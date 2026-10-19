# 03_matmul：从正确性到 Autotune 的完整性能优化推理链

> 重写日期：2026-07-20  
> 核心目标：不仅记录“哪个配置最快”，还要解释为什么提出假设、如何隔离变量、数据证明了什么，以及哪些底层原因仍需 profiler 验证。

## 0. 一句话总览

本实验从一个正确但固定配置的 tiled MatMul 出发，依次研究输出 tile、program 排布、K 分块和 warp 数，再把受控实验筛出的候选交给 autotune。实验最终证明：MatMul 性能来自“数据复用、program 并行度、tile 利用率、片上资源和调度开销”的平衡；更大的 tile、更大的 `BLOCK_K`、更多 warp 或更强的分组都不保证更快，而且单次微秒级 Benchmark 会受到明显运行状态干扰。

完整决策链：

```text
4x4 手算 tiling
-> 实现朴素 kernel
-> correctness 闭环
-> 固定 32x32x32 baseline
-> 输出 tile sweep：发现 512^3 与 2048^3 的性能反转
-> Grouped Ordering：验证执行顺序可能影响性能，也暴露跨次运行不稳定
-> BLOCK_K sweep：找到循环开销与片上资源压力的中间点
-> num_warps sweep：确认更多线程不等于更快
-> autotune：让不同 M/N/K 选择整套配置
-> 最终多 shape Benchmark：验证收益与限制
```

## 1. 算子定义与手算示例

### 1.1 数学定义

二维矩阵乘法：

```text
A[M, K] @ B[K, N] -> C[M, N]
```

元素公式：

```text
C[i, j] = sum(k=0...K-1) A[i, k] * B[k, j]
```

三个逻辑维度：

```text
M：C 的行，也是 A 的行
N：C 的列，也是 B 的列
K：A 的列和 B 的行，是归约维度
```

`512x512x512` 是 `(M,N,K)`，不是三维矩阵。最后一个 512 表示每个 C 元素需要沿 K 累加 512 个乘积项。

### 1.2 4x4、BLOCK=2 的手算 tiling

设：

```text
M=N=K=4
BLOCK_M=BLOCK_N=BLOCK_K=2
```

每个 program 计算 C 的一个 `2x2` 输出 tile。计算右上角 `C[0:2, 2:4]` 时：

```text
第 1 轮 K=0:2：A[0:2, 0:2] @ B[0:2, 2:4]
第 2 轮 K=2:4：A[0:2, 2:4] @ B[2:4, 2:4]
```

两轮局部乘积累加到同一个 `2x2` accumulator。

可复述结论：

```text
C 的 tile 行决定 A 的行；
C 的 tile 列决定 B 的列；
K 循环让 A 向右移动、B 向下移动。
```

## 2. program、tile 与数据范围的映射

### 2.1 program 数量

一个 program 负责一个 `[BLOCK_M, BLOCK_N]` 输出 tile：

```text
num_programs = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)
```

K 不增加 program 数，只决定 program 内部的循环次数：

```text
k_loops = cdiv(K, BLOCK_K)
```

`num_warps` 决定每个 program 使用多少个 warp：

```text
grid       -> 有多少个 program
num_warps  -> 每个 program 有多少个 warp
K loop     -> 每个 program 内部累加多少轮
```

例如 `M=N=K=512, BM=128, BN=64, BK=32, w4`：

```text
program 数 = (512/128) * (512/64) = 32
每个 program 使用 4 warps
每个 program 进行 512/32 = 16 轮 K 循环
```

### 2.2 二维广播生成指针 tile

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
offs_k = tl.arange(0, BLOCK_K)
```

`[:, None]` 把一维向量变成列向量，`[None, :]` 把一维向量变成行向量。通过广播构造二维地址：

```python
a_ptrs = (
    a_ptr
    + offs_m[:, None] * stride_am
    + offs_k[None, :] * stride_ak
)

b_ptrs = (
    b_ptr
    + offs_k[:, None] * stride_bk
    + offs_n[None, :] * stride_bn
)
```

地址公式：

```text
A[m,k] 偏移 = m * stride_am + k * stride_ak
B[k,n] 偏移 = k * stride_bk + n * stride_bn
C[m,n] 偏移 = m * stride_cm + n * stride_cn
```

`tl.load` 接收的是二维指针 tensor；不是 Python 逐元素读取。

## 3. Global Memory、中间状态与数值边界

### 3.1 数据生命周期

每个 program 的逻辑过程：

```text
Global Memory 读取 A tile / B tile
-> tl.dot
-> FP32 accumulator 在 program 生命周期内持续累加
-> K 循环结束
-> accumulator 转换并写回 C tile
```

中间局部乘积不会显式写回 Global Memory。program 之间不能共享寄存器或 SRAM；跨 program 的数据复用主要依赖缓存层次，尤其是 L2。

理想语义下，整个算子至少需要读取 A、读取 B、写入 C。但实际 tiled kernel 中，相同 A/B 数据会被多个 programs 请求；请求最终来自 DRAM 还是由 L2 命中满足，需要硬件计数器才能确认。

### 3.2 accumulator 与数值精度

```python
accumulator = tl.zeros(
    (BLOCK_M, BLOCK_N),
    dtype=tl.float32,
)
```

输入和输出为 FP16，中间使用 FP32 累加，减少长 K 归约的舍入误差。MatMul 不需要 Softmax 那样的 max-shift 稳定化，但仍存在输入精度、`tl.dot` 硬件精度策略和最终 FP16 写回误差。

### 3.3 mask 与 padding

`cdiv` 只扩大启动 grid，不会真的补齐输入内存。边缘 tile 使用 mask：

```python
a = tl.load(a_ptrs, mask=a_mask, other=0.0)
b = tl.load(b_ptrs, mask=b_mask, other=0.0)
tl.store(c_ptrs, accumulator, mask=c_mask)
```

越界输入补 0，不改变乘加结果；store mask 阻止越界写回。

边缘 tile 的线程仍可能执行部分无效 lane，因此非对齐 shape 会产生 padding 开销和 tile 利用率下降。本文主要参数 sweep 使用 512/2048 等整除 shape，目的是先排除 padding 这一混杂因素；correctness 测试再专门覆盖非对齐边界。

## 4. Correctness 闭环

手动 kernel：

| Shape `(M,N,K)` | 覆盖目标 |
| --- | --- |
| `32,32,32` | 对齐基本路径 |
| `64,48,96` | 矩形矩阵、N 尾块 |
| `37,53,29` | M/N/K 全部非对齐 |
| `1,1,1` | 极小矩阵、几乎全 padding |

autotuned kernel：

```text
257x263x129
```

该 shape 验证 autotuner 选择任一候选配置后，三个维度的 mask 仍正确。

参考实现与容差：

```python
expected = torch.matmul(a, b)
torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
```

最终结果：

```text
5 passed
```

## 5. Benchmark 口径与环境

### 5.1 环境

项目已有实验环境记录：

```text
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
SM count: 26
VRAM: 7.96 GiB
Compute Capability: 12.0
PyTorch: 2.11.0+cu128
Triton: 3.6.0
CUDA: 12.8
运行环境: WSL / Laptop GPU
dtype: FP16
```

Laptop 动态频率、功耗状态、温度以及 WSL/桌面调度都可能影响微秒级结果。

### 5.2 计时范围

```text
warmup = 100 ms
rep = 300 ms
quantiles = P20 / P50 / P80
```

主要决策使用 P50；P20/P80 用于判断分布宽度和异常波动。

计时表达式包含：

```text
Python wrapper
+ 输出 torch.empty 分配
+ kernel launch
+ kernel 执行
```

PyTorch 和 Triton 都创建输出，因此它反映用户视角算子延迟；但对十几微秒的小矩阵，wrapper、分配、缓存查询和 launch 会占据较大比例。

JIT 编译、correctness 和首次 autotune 在正式计时前完成，不计入 P20/P50/P80。

### 5.3 TFLOPS 口径

```text
FLOPs = 2 * M * N * K
TFLOPS = FLOPs / latency_ms * 1e-9
```

系数 2 表示一次乘法和一次加法。MatMul 主要使用 TFLOPS，而不是用语义 GB/s 宣称真实 DRAM 带宽。没有 profiler 时，TFLOPS 只说明端到端计算吞吐，不能直接证明 Tensor Core 利用率、L2 命中率或 occupancy。

---

# 6. 实验一：固定 32x32x32 的朴素 baseline

## 6.1 实验问题

先建立一个正确、简单、可重复比较的性能起点：固定小 tile 的 Triton MatMul 随矩阵尺寸增大，会从固定开销区进入稳定计算吞吐区吗？

## 6.2 控制变量

固定：

```text
BM=32, BN=32, BK=32, num_warps=4
FP16 input/output, FP32 accumulator
相同 warmup/rep/quantiles
```

只改变方阵尺寸 `M=N=K=256/512/1024/2048`，从而观察问题规模，而不混入 tile 配置变化。

## 6.3 性能预测

- 256 等小 shape 主要受输出分配和 launch 固定开销限制；
- shape 增大后 program 数增加，GPU 并行度和 `tl.dot` 占比提高；
- 小 `32x32` 输出 tile 的数据复用有限，大矩阵上可能逐渐受 tile 粒度约束；
- 吞吐应整体随规模上升，但单点可能受到运行状态影响。

## 6.4 关键结果

当前 `matmul-naive.csv`：

| Shape | Torch P20/P50/P80 ms | Triton P20/P50/P80 ms | Torch TFLOPS | Triton TFLOPS | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| `256^3` | `0.0130/0.0150/0.0161` | `0.0116/0.0146/0.0152` | 2.23 | 2.29 | 1.028x |
| `512^3` | `0.0214/0.0240/0.0253` | `0.0533/0.0612/0.0648` | 11.17 | 4.39 | 0.393x |
| `1024^3` | `0.0987/0.1048/0.1173` | `0.1184/0.1210/0.1250` | 20.49 | 17.74 | 0.866x |
| `2048^3` | `0.8964/1.0377/1.1770` | `0.8780/0.8911/0.9491` | 16.56 | 19.28 | 1.165x |

## 6.5 现象总结

- 256 的两者延迟都只有约 15 微秒，属于固定开销显著的区域；
- 512 出现异常性能台阶：Triton 比相邻趋势预期慢很多；
- 1024 接近但仍落后；
- 2048 的 Triton P50 反而领先，但 PyTorch 分位区间很宽；
- 曲线不是简单单调平台，存在运行状态和配置适配共同影响。

## 6.6 原因解释

直接可建立的因果链：

```text
shape 增大
-> program 数增加
-> 固定开销占比下降
-> 计算吞吐逐渐成为主要因素
```

`32x32` tile 每个 program 的工作较小，可能产生较多 program 调度与重复 A/B 请求。512 的异常不能仅由 program 数解释，因为后续相同配置曾测到约 `0.0235 ms`，而当前 CSV 为 `0.0612 ms`。

## 6.7 证据边界

数据直接支持：同一配置跨 shape 表现差异大；512 单点不稳定；小 shape 固定开销不可忽略。

数据不能证明：512 一定发生 cache miss、spill、某种 Tensor Core 路径切换或 occupancy 降低。确认这些原因需要 Nsight Compute 的 kernel 名、寄存器、shared memory、L2 hit rate 和 active warps 指标。

## 6.8 当前最优解

baseline 不用于给出最终配置，只作为后续受控实验的比较起点。当前决策是：不要对 512 异常做故事化归因，下一步直接扫描输出 tile。

## 6.9 失败实验

失败 heuristic：看到第一次 `512^3` 只有约 4.4 TFLOPS，就认为该 shape 对 `32x32` tile 存在稳定结构性退化。

修正：同配置独立复测曾达到约 11.4 TFLOPS，说明单次异常不足以建立结构性结论。性能模型必须加入“重复运行和分位数”这一证据要求。

## 6.10 面试表达

> 我先用 `BM=BN=BK=32` 建立 FP16 MatMul baseline，固定 kernel 配置，只改变方阵规模，并用 P20/P50/P80 和 TFLOPS 观察固定开销到计算吞吐的变化。结果中 512 点出现过 4.4 到 11.4 TFLOPS 的跨次波动，因此我没有把它直接归因于 cache 或 occupancy，而是把它视为运行状态敏感点，随后通过单变量 tile sweep 隔离配置影响。这个实验的决策不是选最优参数，而是建立可信的计时和证据边界。

---

# 7. 实验二：输出 tile sweep 与 512/2048 性能反转

## 7.1 实验问题

更大的 `BLOCK_M/BLOCK_N` 能让 A 数据服务更多输出列、B 数据服务更多输出行，提高 tile 内复用；但它也减少 program 数并扩大 accumulator。实验要验证：数据复用收益在什么规模下能超过并行度和片上资源代价？

## 7.2 控制变量

固定：

```text
BLOCK_K=32
GROUP_SIZE_M=1
FP16 / FP32 accumulator
方阵且所有维度整除 tile，排除 padding
```

只改变输出 tile 和配套 warp：

```text
32x32,w4
64x64,w4
128x64,w4
64x128,w4
128x128,w8
```

分别在 `512^3` 和 `2048^3` 上运行，用两个规模隔离“program 是否足够多”的影响。

## 7.3 性能预测

- 大 tile 提高算术强度和 A/B tile 内复用；
- 512 上大 tile 会把 program 数降到 16~64，可能并行度不足；
- 2048 上即使大 tile 仍有 256~1024 个 programs，更可能体现复用收益；
- `128x128` accumulator 最大，理论复用最高，但不一定最快；
- `128x64` 与 `64x128` 理论工作量相同，若性能不同则说明硬件布局或调度存在方向敏感性。

## 7.4 关键结果

### 512^3

Torch：`P20/P50/P80 = 0.0233/0.0256/0.0274 ms`，10.49 TFLOPS。

| BMxBN,warps | programs | Triton P20/P50/P80 ms | TFLOPS | vs Torch |
| --- | ---: | --- | ---: | ---: |
| `32x32,w4` | 256 | `0.0251/0.0276/0.0294` | 9.74 | 0.928x |
| `64x64,w4` | 64 | `0.0276/0.0310/0.0321` | 8.65 | 0.824x |
| `128x64,w4` | 32 | `0.0268/0.0279/0.0307` | 9.63 | 0.918x |
| `64x128,w4` | 32 | `0.0249/0.0269/0.0275` | 9.96 | 0.950x |
| `128x128,w8` | 16 | `0.0372/0.0636/0.0756` | 4.22 | 0.402x |

### 2048^3

Torch：`P20/P50/P80 = 0.7257/0.7295/0.7336 ms`，23.55 TFLOPS。

| BMxBN,warps | programs | Triton P20/P50/P80 ms | TFLOPS | vs Torch |
| --- | ---: | --- | ---: | ---: |
| `32x32,w4` | 4096 | `0.9277/1.0066/1.6540` | 17.07 | 0.725x |
| `64x64,w4` | 1024 | `0.7717/0.8348/0.9290` | 20.58 | 0.874x |
| `128x64,w4` | 512 | `0.6348/0.6422/0.6489` | 26.75 | 1.136x |
| `64x128,w4` | 512 | `0.6465/0.7530/0.8663` | 22.82 | 0.969x |
| `128x128,w8` | 256 | `0.6861/0.7312/0.8003` | 23.50 | 0.998x |

## 7.5 现象总结

- 512 上大 tile 没有形成单调收益，`128x128` 出现明显退化和宽分位区间；
- 2048 上性能随 tile 从 32 增至 128x64 形成清晰台阶；
- 同一个 `128x64` 在 512 上没有优势，在 2048 上成为最佳，出现规模反转；
- `128x64` 明显优于 `64x128`，尽管两者 program 数和输出面积相同；
- `128x128` 理论复用最高，但没有超过 `128x64`。

## 7.6 原因解释

简化 FP16 算术强度：

```text
AI ~= BLOCK_M * BLOCK_N / (BLOCK_M + BLOCK_N) FLOP/byte
```

```text
32x32   -> 16.0
64x64   -> 32.0
128x64  -> 42.7
128x128 -> 64.0
```

512 上：

```text
tile 增大
-> program 数从 256 降至 16~64
-> 可调度并行工作减少
-> accumulator 和 program 资源需求增加
-> 复用收益不足以抵消并行度/资源代价
```

2048 上：

```text
大 tile 后仍有 256~512+ programs
-> program 并行度仍充足
-> A/B tile 内复用和更高计算粒度开始主导
-> 128x64 获得最佳平衡
```

## 7.7 证据边界

数据直接支持：存在规模反转；大 tile 不是普适最优；`128x64` 与 `64x128` 存在稳定方向差异；理论 AI 最高不等于实际最快。

数据不能独立证明：`128x128` 一定发生 register spill，或 `128x64` 一定具有更高 L2 hit rate。逻辑 accumulator 更大是确定事实，但是否 spill、occupancy 降到多少必须由 profiler 验证。

## 7.8 当前最优解

在 `2048^3, BK=32, group=1` 条件下选择 `BM=128, BN=64, w4`；在 512 上则保留更小 tile。该结论解决了固定 tile 不适配不同规模的问题，但无法解决跨 shape 自动选择，因此后续仍需 autotune。

## 7.9 失败实验

失败 heuristic 1：tile 越大，算术强度越高，所以一定越快。`128x128` 被数据推翻。

失败 heuristic 2：交换 M/N 不改变理论 FLOPs 和输出面积，所以 `128x64` 与 `64x128` 应相同。数据表明硬件执行并不对称，但具体原因尚未由 profiler 定位。

## 7.10 面试表达

> 我用相同 BK、group、dtype 和整除 shape，只扫描输出 tile，并同时测试 512 和 2048。预测是大 tile 提高数据复用，但会减少 program 数和增加 accumulator 压力。结果在 512 上大 tile 退化，而在 2048 上 `128x64` 从 17 TFLOPS 级提升到 26.75 TFLOPS，出现规模反转；`128x128` 虽然理论 AI 最高却更慢。因此我选择 128x64 作为大矩阵候选，但不把退化直接断言为 spill，而是把寄存器和 occupancy 留给 profiler 验证。

---

# 8. 实验三：Grouped Ordering 与 L2 局部性

## 8.1 实验问题

不改变计算量、program 数和 tile，只改变线性 pid 到 `(pid_m,pid_n)` 的顺序，能否让相邻 programs 更容易复用相同 A/B tile，从而降低实际 Global Memory 访问代价？

## 8.2 控制变量

固定：

```text
M=N=K=2048
BM=128, BN=64, BK=32, num_warps=4
```

只改变：

```text
GROUP_SIZE_M = 1, 2, 4, 8, 16
```

数学结果、program 数、每个 program 的 `tl.load/tl.dot` 结构保持不变，因此差异主要来自执行顺序和运行状态。

## 8.3 性能预测

- group 1 主要连续遍历 N，A 复用较直接，B 的再次使用间隔较长；
- 适度增加 group，可让多个 M tiles 连续使用同一 B tile，提高 L2 局部性概率；
- group 太大会扩大同时活跃的 A 工作集，收益可能饱和或反转；
-如果工作集已适配缓存，group 差异可能很小。

## 8.4 关键结果

本实验出现了互相冲突的独立运行，必须同时保留。

### 早期独立运行（截图记录）

| Group | P50 ms | TFLOPS | vs Group1 |
| ---: | ---: | ---: | ---: |
| 1 | 0.6746 | 25.47 | 1.000x |
| 2 | 0.5943 | 28.91 | 1.135x |
| 4 | 0.6571 | 26.15 | 1.027x |
| 8 | 0.6559 | 26.19 | 1.029x |
| 16 | 0.6585 | 26.09 | 1.024x |

用户随后多次独立运行观察到 group 2 明显高于其他组。

### 当前落盘 `matmul-group-sweep-2048.csv`

Torch：`0.6192/0.6233/0.6383 ms`，27.56 TFLOPS。

| Group | Triton P20/P50/P80 ms | TFLOPS | vs Group1 |
| ---: | --- | ---: | ---: |
| 1 | `0.5969/0.6039/0.6250` | 28.45 | 1.000x |
| 2 | `0.8061/0.9062/0.9143` | 18.96 | 0.666x |
| 4 | `0.6045/0.6092/0.6234` | 28.20 | 0.991x |
| 8 | `0.6277/0.7108/0.7883` | 24.17 | 0.850x |
| 16 | `0.7008/0.7714/0.8109` | 22.27 | 0.783x |

## 8.5 现象总结

- 早期多轮观察中 group 2 形成约 13.5% 的明确台阶；
- 当前落盘运行中 group 2 反而严重退化，group 1/4 最好；
- group 8/16 在当前运行中呈随 group 增大而下降的趋势；
- 访问顺序确实可能影响性能，但“group 2 稳定全局最优”没有被所有落盘数据复现。

## 8.6 原因解释

Grouped Ordering 的确定机制是：让共享 B tile 的多个 M programs 更靠近，同时让少量 A tile 在较短时间内重复出现。它不减少 load 指令，只可能让部分 load 从 L2 满足而不是访问更远层次。

合理假设链：

```text
适度 grouping
-> 相邻 program 共享 B 数据
-> L2 局部性概率提高
-> DRAM 请求可能减少
```

但当前跨次冲突说明，GPU 动态状态、测量顺序、缓存初始状态或其他未控制因素足以覆盖这一收益。

## 8.7 证据边界

数据直接支持：不同 group 会改变性能；group 过大在当前落盘运行中不利；跨次结果存在显著冲突。

数据不能证明：某次 group 2 快就是 L2 hit rate 提升，或某次 group 2 慢就是 cache thrashing。没有硬件计数器时，只能称为“与缓存局部性假设一致/不一致”，不能断言具体 cache miss 数。

要收紧证据，需要：

```text
随机化 group 测量顺序
多轮独立进程重复
记录 GPU 时钟/功耗/温度
Nsight Compute 比较 L2 hit rate、DRAM bytes 和 occupancy
```

## 8.8 当前最优解

手动候选曾采用 `GROUP_SIZE_M=2`，因为早期多轮独立测试中它持续领先；但新版证据链不把它写成普适结论。最终接口使用 autotune，让 group 与其他参数作为整套配置共同选择。

## 8.9 失败实验

失败 heuristic：Grouped Ordering 是标准优化，所以 `GROUP_SIZE_M=2/8` 必然比 1 快。

当前 CSV 直接推翻了“必然”二字。性能模型修正为：grouping 只改变数据访问顺序，其收益依赖 shape、tile、硬件缓存和运行状态，必须重复测量并最好用 profiler 验证。

## 8.10 面试表达

> 我固定 2048 方阵、128x64x32 tile 和 4 warps，只改变 GROUP_SIZE_M，因此计算量和 program 数完全不变，实验只研究 program 顺序。早期多轮中 group 2 比 group 1 快约 13.5%，但后续整套脚本复跑时 group 2 反而退化，说明访问顺序影响真实存在，但单靠计时无法稳定证明 L2 机制。我因此保留 grouping 作为 autotune 维度，而不硬编码“group 2 永远最优”，并把 L2 hit rate 归因留给 profiler。

---

# 9. 实验四：BLOCK_K 与归约粒度

## 9.1 实验问题

增大 `BLOCK_K` 会减少 K 循环、指针更新和 dot 调度次数，但每轮 A/B operand tile 更大。实验要寻找循环开销和片上资源需求的平衡点。

## 9.2 控制变量

固定：

```text
M=N=K=2048
BM=128, BN=64
GROUP_SIZE_M=2
num_warps=4
```

只改变：

```text
BLOCK_K = 16, 32, 64, 128
```

输出 tile、program 数和 accumulator 形状不变，因此主要隔离 K 循环宽度和 operand 资源需求。

## 9.3 性能预测

- BK16 循环 128 次，循环和调度开销较高；
- BK32/64 应逐步摊薄循环开销；
- BK128 只有 16 轮，但单轮 operand tile 很大，可能增加 shared memory/register 使用并降低 occupancy；
- 因此预计存在中间最优点，而不是单调随 BK 增长。

## 9.4 关键结果

当前落盘 CSV：Torch `0.6680/0.6721/0.6768 ms`，25.56 TFLOPS。

| BK | K loops | Triton P20/P50/P80 ms | TFLOPS | vs Torch |
| ---: | ---: | --- | ---: | ---: |
| 16 | 128 | `0.6716/0.6755/0.7267` | 25.43 | 0.995x |
| 32 | 64 | `0.7444/0.8751/0.9163` | 19.63 | 0.768x |
| 64 | 32 | `0.6500/0.6588/0.6655` | 26.08 | 1.020x |
| 128 | 16 | `0.7434/0.8216/0.9283` | 20.91 | 0.818x |

早期独立运行中：

```text
BK32 P50 = 0.6450 ms
BK64 P50 = 0.6033 ms
BK64 相对 BK32 提升约 6.9%
```

两个运行都指向 BK64 最佳，但 BK32 的绝对值和提升幅度波动很大。

## 9.5 现象总结

- BK64 在早期和当前落盘结果中都位于最佳或第一梯队；
- BK128 没有因循环最少而最快，形成明显反转；
- 当前 BK32 是异常低点，P20/P80 区间也很宽；
- BK16 相对稳定但没有超过 BK64。

## 9.6 原因解释

BK16：

```text
K 循环多
-> load/dot/指针更新与循环控制次数多
-> 固定调度成本难以摊薄
```

BK64：

```text
循环次数降至 32
-> 单轮计算量增加
-> 循环开销被摊薄
-> operand tile 仍处于当前硬件可接受范围
```

BK128 的每轮逻辑 operand 数据：

```text
A[128,128] FP16 = 32 KiB
B[128,64]  FP16 = 16 KiB
合计 = 48 KiB/stage
```

如果编译器进行多级流水，可能需要多份缓冲；因此更大的 BK 会增加片上资源需求。这能解释“为什么可能反转”，但不能证明具体发生了 spill 或 occupancy 降低。

## 9.7 证据边界

数据直接支持：BK64 优于 BK128；循环更少不等于更快；BK32 在不同运行中不稳定。

数据不能证明：BK128 一定发生 shared-memory bottleneck 或 register spill。需要读取编译后的资源使用量和 profiler 指标。

## 9.8 当前最优解

选择 `BLOCK_K=64` 作为手动和 autotune 的关键候选。它解决了 BK16 循环过细和 BK128 tile 过大的两端问题，但不保证所有 K 长度和 GPU 都最优。

## 9.9 失败实验

失败 heuristic：BK 越大，K 循环越少，所以性能越好。BK128 被数据明确推翻。

修正后的模型：BK 只是在循环开销和片上 operand/流水资源之间做权衡；理论算术强度对 BK 的简化表达近似不变，性能差异来自执行组织而不是 FLOPs 变化。

## 9.10 面试表达

> 我固定输出 tile、group、warp 和 2048 方阵，只扫描 BK，因此 program 数和 accumulator 不变。预测小 BK 会有过多 K 循环，大 BK 会提高片上 operand 和流水缓冲压力。结果 BK64 在独立运行和当前 CSV 中都最好，而 BK128 明显反转。于是我把 BK64 放入最终候选，但没有把 BK128 的退化直接称为 spill，因为那需要编译资源或 Nsight 证据。

---

# 10. 实验五：num_warps 与 program 内并行

## 10.1 实验问题

同一个输出 tile 由多少 warp 协作最合适？更多 warp 能分摊工作，但也增加线程和资源占用，可能降低 resident programs。

## 10.2 控制变量

固定：

```text
M=N=K=2048
BM=128, BN=64, BK=64
GROUP_SIZE_M=2
```

只改变：

```text
num_warps = 2, 4, 8
```

program 数、tile、K 循环和数学计算量不变。

## 10.3 性能预测

- 2 warps 资源占用低，但可能不足以隐藏延迟或充分协作执行大 tile；
- 4 warps 可能在 program 内并行和 occupancy 之间平衡；
- 8 warps 不增加 program 数，只增加每个 program 的线程，可能出现收益饱和或下降。

## 10.4 关键结果

Torch：`0.6209/0.6250/0.6378 ms`，27.49 TFLOPS。

| warps | threads/program | Triton P20/P50/P80 ms | TFLOPS | vs w4 | vs Torch |
| ---: | ---: | --- | ---: | ---: | ---: |
| 2 | 64 | `0.6082/0.6112/0.6184` | 28.11 | 0.992x | 1.022x |
| 4 | 128 | `0.5992/0.6062/0.6290` | 28.34 | 1.000x | 1.031x |
| 8 | 256 | `0.6177/0.6440/0.7174` | 26.68 | 0.941x | 0.970x |

## 10.5 现象总结

- w2 与 w4 处于同一平台，P50 只差约 0.8%；
- w4 略优，但差距可能落在运行噪声范围；
- w8 出现清晰下降和更宽 P20/P80 区间；
- 更多线程没有带来更高吞吐。

## 10.6 原因解释

```text
warps 增加
-> 每个 program 的协作线程增加
-> 单线程承担的逻辑工作可能下降
-> 但 program 的线程/寄存器/调度资源需求增加
-> 同时驻留 program 数可能下降
```

w2/w4 相近说明该 tile 在 64~128 线程范围内已能有效执行；w8 的额外并行没有覆盖资源和调度代价。

## 10.7 证据边界

数据直接支持：w8 比 w2/w4 慢；w2/w4 性能接近。

数据不能证明：w8 具体把 occupancy 降到了多少，或是寄存器、warp 调度还是 Tensor Core 映射主导。必须通过 profiler 或编译元数据确认。

## 10.8 当前最优解

选择 `num_warps=4` 作为手动候选，因为它 P50 最低且是常见平衡点；同时保留 w2 作为可能的 autotune 候选。它不能解决 shape 变化导致的整体配置失配。

## 10.9 失败实验

失败 heuristic：大 tile 应该派更多 warp，8 warps 会更快。数据表明 8 warps 反而慢约 5.9%。

修正：warp 数是 program 内并行度，不是全局 program 数；更多 warp 会消耗更多 SM 资源，必须与 tile 和 occupancy 联合考虑。

## 10.10 面试表达

> 我固定 128x64x64 tile、group 2 和 2048 方阵，只扫描 2/4/8 warps。预测更多 warp 能提升 program 内并行，但会增加资源压力。结果 2 和 4 warps 基本持平，4 warps P50 最低，而 8 warps 下降到 26.68 TFLOPS且波动更宽。因此我选择 w4，但只把 w8 退化归因到“资源/调度权衡”，不在没有 profiler 时断言具体 occupancy 或 spill。

---

# 11. 实验六：Autotune 与最终多 shape 验证

## 11.1 实验问题

前面逐个参数测试得到的是局部候选，参数之间存在交互，而且方阵最优配置可能不适合矩形矩阵。能否让 Triton 针对 `(M,N,K)` 自动选择整套配置，并在多种 shape 上比固定手动配置更稳健？

## 11.2 控制变量

比较三种 provider：

```text
torch.matmul
手动候选：BM128/BN64/BK64/group2/w4
matmul_autotuned
```

所有 provider 使用相同随机 FP16 输入、相同输出语义、相同 correctness 容差和相同计时参数。

autotune key：

```python
key=["M", "N", "K"]
```

候选同时包含小/大 tile、BK32/64、group1/2、warps4/8、`num_stages=2/3/4`。首次正确性调用完成编译和选型，不进入正式计时。

## 11.3 性能预测

- 256 小矩阵的调度与缓存查询开销可能超过自动选型收益；
- 512/1024 等中等 shape 更可能从特化配置受益；
- 2048 方阵上手动候选已经较强，autotune 可能只小幅提升；
- 矩形 shape 会暴露固定 `128x64` 方向和 program grid 的过拟合；
- autotune 不会自动与 PyTorch 比较，因此不保证所有 shape 超过供应商库。

## 11.4 关键结果

当前落盘 `matmul-final.csv`：

| Shape | Torch P20/P50/P80 | Manual P20/P50/P80 | Auto P20/P50/P80 | Torch/Manual/Auto TFLOPS | Manual x | Auto x |
| --- | --- | --- | --- | --- | ---: | ---: |
| `256^3` | `0.0110/0.0140/0.0151` | `0.0126/0.0148/0.0167` | `0.0142/0.0199/0.0332` | `2.39/2.27/1.69` | 0.952x | 0.707x |
| `512^3` | `0.0305/0.0429/0.0506` | `0.0273/0.0290/0.0311` | `0.0269/0.0289/0.0315` | `6.25/9.26/9.28` | 1.481x | 1.485x |
| `1024^3` | `0.1063/0.1175/0.1336` | `0.1267/0.1358/0.1474` | `0.1180/0.1204/0.1225` | `18.28/15.81/17.84` | 0.865x | 0.976x |
| `2048^3` | `0.7363/0.7705/0.7955` | `0.7364/0.7440/0.7481` | `0.6511/0.6611/0.6917` | `22.30/23.09/25.99` | 1.036x | 1.165x |
| `512x1024x256` | `0.0212/0.0212/0.0243` | `0.0210/0.0228/0.0252` | `0.0208/0.0215/0.0249` | `12.63/11.75/12.47` | 0.930x | 0.987x |
| `1024x512x2048` | `0.1017/0.1049/0.1055` | `0.1055/0.1085/0.1121` | `0.0908/0.0951/0.0984` | `20.48/19.78/22.58` | 0.966x | 1.103x |
| `2048x1024x512` | `0.0923/0.0928/0.0948` | `0.0916/0.0924/0.0942` | `0.0870/0.0892/0.0904` | `23.13/23.25/24.08` | 1.005x | 1.041x |

另一次最终运行曾得到：

```text
Auto speedup：0.715x, 1.468x, 1.338x, 0.984x, 1.094x, 1.055x, 1.083x
```

当前落盘运行为：

```text
Auto speedup：0.707x, 1.485x, 0.976x, 1.165x, 0.987x, 1.103x, 1.041x
```

## 11.5 现象总结

- 256 上 autotune 稳定落后，固定开销和选型噪声占主导；
- 512 上两次运行都显示约 1.47~1.49x 的明显优势；
- 1024 和 2048 的领先关系在不同运行间发生反转；
- `1024x512x2048` 两次都由 autotune 领先，说明矩形 shape 能从配置选择中受益；
- autotune 通常优于手动候选，但不是每个 shape 都优于 PyTorch；
- 结果存在性能台阶和跨次差异，不能用一张图宣称普适胜出。

## 11.6 原因解释

```text
M/N/K 改变
-> program grid、边缘 tile 和并行度改变
-> A/B 数据复用方向改变
-> 合适的 BM/BN/BK/group/warps/stages 组合改变
```

autotune 通过对完整配置组合计时，处理单变量实验无法覆盖的参数交互。矩形矩阵中固定 `128x64` 可能在某一方向产生较差 tile/grid 适配，而 autotune 可以换成其他候选。

256 的 kernel 本体太短，`matmul_autotuned` 的 Python dispatch、缓存查询和输出分配相对占比更大；即使内部 kernel 配置合理，端到端 P50 也可能更慢。

## 11.7 证据边界

数据直接支持：autotune 能在部分中等和矩形 shape 上显著超过手动候选；256 不适合承担 autotune 开销；不同运行状态会改变部分 shape 的胜负。

数据不能证明：autotune 选中的具体配置为什么更快，因为当前结果没有持久化每个 key 的 winner，也没有 profiler 指标。不能从速度差直接断言某候选使用了特定 Tensor Core 指令或获得了更高 occupancy。

最终 Benchmark 测的是 wrapper 级端到端延迟，不是预分配输出后的纯 kernel time。

## 11.8 当前最优解

接口层面推荐：

```text
中等/大规模和多 shape 场景 -> matmul_autotuned
极小或固定单一 shape       -> 可考虑手动配置或供应商库
```

手动候选 `128x64x64/group2/w4` 用于解释和对照，不再宣称全局最优。autotune 解决 shape 与参数交互问题，但不能解决候选集不完整、调优开销、运行状态噪声和供应商库竞争。

## 11.9 失败实验

失败 heuristic 1：逐个参数找到的局部最优组合就是全局最优。矩形 shape 中 autotune 相对手动候选的提升推翻了这一点。

失败 heuristic 2：autotune 一定比手动配置和 PyTorch 快。256 和部分 1024/2048 运行推翻了这一点。

失败 heuristic 3：一次最终图即可确定所有 shape 的胜负。两次最终运行中 1024/2048 发生反转，说明必须记录重复性。

## 11.10 面试表达

> 我先通过单变量实验理解 BM/BN、group、BK 和 warp 的作用，再把合理候选组成 `triton.Config`，以 M/N/K 为 key 做 autotune。最终用相同输入比较 PyTorch、手动候选和 autotuned Triton，并在计时前完成 correctness 和调优。autotune 在 512 和部分矩形 shape 上稳定提升，512 达到约 1.48x，但 256 因固定开销明显落后，1024/2048 的胜负还会跨运行变化。因此我的决策是大中 shape 使用 autotune、小 shape 谨慎使用，同时把 winner 配置、L2、occupancy 和指令路径留给后续 profiler 验证。

---

# 12. 从 baseline 到最终版本的决策链

| 阶段 | 观察 | 性能模型修正 | 决策 |
| --- | --- | --- | --- |
| 固定 baseline | 512 单点跨次波动巨大 | 单次 P50 不能证明结构性退化 | 使用分位数和重复实验 |
| 输出 tile | 512 大 tile 退化，2048 大 tile 提升 | 数据复用必须与 program 数、资源压力平衡 | 大矩阵候选选 `128x64` |
| Grouped Ordering | 早期 group2 提升，后续落盘结果冲突 | 排布收益依赖缓存和运行状态 | 不硬编码普适 group，纳入 autotune |
| BLOCK_K | BK64 最佳，BK128 反转 | 减少循环与片上 operand 资源存在平衡 | 保留 BK64 |
| num_warps | w2/w4 平台，w8 下降 | program 内线程越多不一定越快 | 保留 w4，候选可含 w2 |
| autotune | 矩形 shape 优于固定配置，小 shape 退化 | 参数存在交互，调优也有固定开销 | 大中 shape 自动选型，小 shape 谨慎 |

最终核心结论：

```text
MatMul 的最优配置由 shape、硬件和参数交互共同决定。
先用受控实验建立因果模型，再用 autotune 选择整套配置；
没有重复运行和 profiler 时，不把计时现象包装成确定的底层事实。
```

# 13. 当前支持范围与限制

已经覆盖：

- CUDA FP16 二维 MatMul；
- FP32 accumulator；
- 显式 stride 地址计算；
- M/N/K 非 block 整数倍；
- Grouped Ordering；
- 手动配置与 M/N/K keyed autotune；
- 方阵和矩形矩阵 correctness、P20/P50/P80、TFLOPS。

仍未完整覆盖：

- BF16、FP32/TF32 精度策略；
- batch/head 维度；
- 转置和任意非连续输入的系统测试；
- bias/activation epilogue fusion；
- backward；
- winner 配置持久化；
- 随机化实验顺序与 GPU 时钟/功耗记录；
- Nsight 对 L2 hit rate、DRAM bytes、Tensor Core 指令、occupancy、寄存器和 spill 的验证；
- persistent MatMul、TMA、warp specialization。

# 14. 后续学习路线

严格按项目路线继续：

```text
04_layernorm
-> reduction + elementwise fusion
-> 均值/方差数值稳定性
-> 一次读写与 register pressure

05_attention_naive
-> 显式 QK^T、Softmax、P@V
-> scale、mask、shape 与中间张量生命周期
-> 建立 Attention correctness baseline

06_flash_attention_forward
-> tiled QK^T
-> Online Softmax 的 m_i/l_i 重标定
-> 直接累加 P tile @ V tile
-> 不物化完整 attention matrix
```

MatMul 阶段已经提供 FlashAttention 所需的 tiled dot、accumulator、program 排布和性能分析方法，但下一步是 `04_layernorm`，不能跳过既定路线。

# 15. 总体面试回答

> 我从一个 FP16、FP32 累加的 Triton tiled MatMul baseline 开始，先用非对齐 shape 验证 M/N/K mask 和 K 分块正确性，再通过 P20/P50/P80 与 TFLOPS 做受控优化。输出 tile 实验发现 512 上大 tile 因 program 数减少而退化，但 2048 上 `128x64` 因数据复用提高而形成性能台阶；这推翻了“tile 越大越快”。随后我只改变 program 排布测试 Grouped Ordering，观察到分组可能带来两位数收益，但跨次结果冲突，所以没有在缺少 L2 profiler 证据时断言 cache 机制。BLOCK_K 实验表明 BK64 在循环开销和片上 operand 资源之间最好，BK128 推翻了“循环越少越快”；warp 实验则表明 4 warps 与 2 warps 接近、8 warps 反而更慢。最后我把合理组合交给以 M/N/K 为 key 的 autotune，在 512 和多个矩形 shape 上超过手动配置或 PyTorch，但小矩阵因固定开销落后，部分大 shape 还存在跨运行反转。因此最终决策是：用受控实验理解机制，用 autotune 处理参数交互，用重复测量约束结论，并把 spill、L2 miss、occupancy 和具体指令路径留给 Nsight 验证。

# 16. 文件索引

| 文件 | 内容 |
| --- | --- |
| [`kernel.py`](./kernel.py) | 手动 kernel、Grouped Ordering、autotuned launcher |
| [`test_correctness.py`](./test_correctness.py) | 5 组 correctness 测试 |
| [`benchmark.py`](./benchmark.py) | baseline 与四类单变量 sweep |
| [`benchmark_final.py`](./benchmark_final.py) | PyTorch / manual / autotuned 最终对比 |
| `benchmark_results/*.csv` | 当前落盘原始分位数与吞吐数据 |
| `benchmark_results/*.png` | 对应可视化 |

# 17. 参考资料

- Triton Matrix Multiplication Tutorial：<https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html>
- `triton.Config` API：<https://triton-lang.org/main/python-api/generated/triton.Config.html>
