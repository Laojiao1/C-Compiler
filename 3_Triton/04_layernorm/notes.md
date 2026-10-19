# 04_layernorm 复盘

## 1. 阶段目标与最终产物

本阶段实现二维 CUDA Tensor 的 LayerNorm forward：

```text
X[M, N] -> Y[M, N]
```

重点不是只把结果算对，而是完成以下学习闭环：

```text
均值/方差 reduction
-> 数值稳定性
-> normalize + affine elementwise fusion
-> 一次主输入读取与一次输出写回
-> register pressure 假设
-> 不同特征维度 Benchmark
-> num_warps 单变量实验
-> autotune
```

目录中的核心产物：

| 文件 | 用途 |
| --- | --- |
| [`kernel.py`](./kernel.py) | LayerNorm kernel、manual/heuristic/autotuned launcher |
| [`test_correctness.py`](./test_correctness.py) | dtype、mask、数值稳定性和输入契约测试 |
| [`benchmark.py`](./benchmark.py) | 边界、固定总元素、warp sweep 和三方 Benchmark |
| [`benchmark_results/`](./benchmark_results/) | 最终 CSV 和 PNG |

最终正确性结果：

```text
30 passed
```

---

## 2. LayerNorm 数学定义

对一行输入 `x[0:N]`：

```text
mean     = sum(x) / N
variance = sum((x - mean)^2) / N
rstd     = 1 / sqrt(variance + eps)
y        = (x - mean) * rstd * weight + bias
```

这里方差除以 `N`，即总体方差，与 PyTorch LayerNorm 的 `unbiased=False` 行为一致。

如果一整行都是常数 `c`：

```text
mean       = c
variance   = 0
normalized = 0
y          = bias
```

`eps` 防止零方差时出现除零；在有限精度实现中，最终结果允许与理论 `bias` 存在极小舍入误差。

---

## 3. Triton program 映射

当前实现使用 row-wise 映射：

```text
grid = (M,)
一个 program 负责一整行 N 个特征
```

第 `row_idx` 行的起始地址：

```text
x_row_ptr = x_ptr + row_idx * x_row_stride
```

program 内部构造：

```text
offsets = [0, 1, ..., BLOCK_SIZE - 1]
mask    = offsets < N
```

其中：

```text
BLOCK_SIZE = next_power_of_2(N)
```

`BLOCK_SIZE` 是一个 program 处理的逻辑 blocked tensor 宽度，不等于 CUDA thread 数量。逻辑位置最终如何映射到线程、warp、寄存器和 shared memory，由 Triton 编译器决定。

---

## 4. 两次 Reduction 与 Padding 陷阱

第一步求均值：

```python
x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
x = x.to(tl.float32)
mean = tl.sum(x, axis=0) / N
```

第二步求方差：

```python
centered = tl.where(mask, x - mean, 0.0)
variance = tl.sum(centered * centered, axis=0) / N
```

关键陷阱是：即使 padding 在加载时填成 0，也不能直接写：

```python
centered = x - mean
```

因为无效位置会从 `0` 变成 `-mean`，随后其平方 `mean^2` 会进入方差 reduction，污染整行的方差标量。

因此 padding 必须在中心化之后再次清零，并且均值、方差都除以真实 `N`，不能除以 `BLOCK_SIZE`。

---

## 5. 数值稳定性

不采用以下公式：

```text
variance = E[x^2] - E[x]^2
```

当输入具有“大公共偏置 + 小波动”时，`E[x^2]` 与 `E[x]^2` 都很大，而真实方差很小，两个大数相减会产生严重消减误差，甚至得到负方差。

当前实现采用两遍中心化算法：

```text
先求 mean
-> 再求 mean((x - mean)^2)
```

同时将 FP16/BF16 输入提升到 FP32 执行 reduction，避免低精度求和和平方过早损失有效位。

数值压力测试使用 FP32 的大公共偏置和小波动，并与 `torch.nn.functional.layer_norm` 比较。当前实现通过测试，但仍可能因 FP32 mean reduction 顺序与 PyTorch 不同而出现约 `1e-2` 量级的极端输出差异。

更稳定的 Welford 均值/方差联合更新留作后续实验；当前结果不能表述为与 PyTorch 逐 bit 相等。

---

## 6. Elementwise Fusion 与一次读写

拆分实现可能形成多个阶段：

```text
mean
-> centered
-> squared
-> variance
-> normalized
-> affine output
```

如果阶段之间分别启动 kernel，就可能把 `centered`、`squared`、`normalized` 等中间张量反复写入和读出 Global Memory。

当前 Triton kernel 在一个 program 中融合：

```text
读取 x
-> mean reduction
-> variance reduction
-> normalize
-> 乘 weight
-> 加 bias
-> 写回 y
```

因此从源码和算法结构看，主输入 `x` 只有一次 `tl.load`，输出 `y` 只有一次 `tl.store`，不物化全局中间张量。

但“源码级一次 load”不保证物理上绝对只有一次 DRAM 读取。特征维度过大时，编译器可能因片上资源不足发生 spill；是否发生 spill 需要 profiler 或编译元数据验证。

---

## 7. 显存流量与性能指标

设数据类型每元素占 `s` Bytes。

核心张量口径：

```text
读取 x + 写回 y = 2MNs Bytes
```

源码级逻辑口径：

```text
读取 x/weight/bias + 写回 y = 4MNs Bytes
```

Benchmark 主表使用：

```text
Core Effective GB/s = 2MNs / latency
```

原因是 `weight/bias` 会被所有行重复使用，可能大量命中 L2 Cache。`4MNs/time` 可以表达源码级逻辑访问，但不能解释为真实 DRAM 带宽。

同样，`2MNs/time` 也是统一比较指标，不是 profiler 测得的硬件物理流量。

LayerNorm 整体偏 memory-bound，但它比 Vector Add 多了两次 reduction、`rsqrt` 和 affine 计算，因此性能也会受到 reduction 映射和片上资源的影响。

---

## 8. 输入契约与当前支持范围

当前支持：

- 二维 CUDA Tensor；
- 沿最后一维执行 LayerNorm；
- FP16、BF16、FP32；
- `weight/bias` 与输入 dtype、device 一致；
- 最后一维连续；
- `weight/bias` 为连续一维 Tensor，长度等于 `N`；
- `N` 不是 2 的幂时使用 mask；
- 空输入安全返回；
- manual、heuristic 和 autotuned launcher。

当前未覆盖：

- backward；
- 保存 mean/rstd 供 backward 使用；
- 任意前导维度自动展平；
- 任意非连续布局；
- 混合 dtype affine 参数；
- Welford；
- 超过单 program 上限后的分块 reduction。

---

## 9. Correctness 闭环

测试覆盖：

- `(1,1)` 极小输入；
- 普通对齐 shape；
- `N=5/555/2049` 等非 2 的幂；
- FP16、BF16、FP32；
- 常数行与零方差；
- 大公共偏置、小方差；
- `M=0`、`N=0`；
- CPU Tensor；
- 最后一维非连续；
- affine 长度错误；
- dtype 不一致；
- 非法 `eps` 和 `num_warps`；
- 超出单 program 支持范围；
- autotuned launcher 的代表 shape。

浮点 reduction 顺序可能不同，因此根据 dtype 设置 `atol/rtol`，而不是要求逐 bit 相等。

最终结果：

```text
30 passed
```

---

## 10. Benchmark 方法：先保证比较公平，再讨论快慢

### 10.1 Benchmark 要回答的问题

本阶段不是只测一个大 shape，然后报告一个最高 GB/s，而是依次回答三个不同问题：

1. `N` 跨越 2 的幂时，padding 和 reduction 宽度会造成什么性能台阶？
2. 总元素数相同但 `M/N` 比例不同时，LayerNorm 性能是否相同？
3. `num_warps` 对短、中、长 reduction 的影响是否一致？

三个问题分别对应实验一、实验二和实验三。只有将问题拆开并控制变量，才能知道性能变化是由哪一个因素造成的。

### 10.2 对比对象

主表比较三个 provider：

```text
PyTorch baseline
    torch.nn.functional.layer_norm

Triton manual
    固定 num_warps=4，作为可复现的初始实现

Triton autotuned
    在 w1/w2/w4/w8 中按 M/N 测量并缓存最佳候选
```

PyTorch 本身通常也是 fused LayerNorm，因此这个对比不是“融合实现对拆分 eager 实现”，而是两个优化实现之间的算子级比较。

### 10.3 环境与计时范围

```text
GPU：RTX 5060 Laptop GPU
SM：26
dtype：FP16
warmup：100 ms
rep：300 ms
quantiles：P20 / P50 / P80
```

计时范围是用户视角的算子级延迟：

```text
Python wrapper
+ 输出 Tensor 分配
+ kernel launch
+ kernel 执行
```

输入、weight 和 bias 在计时区间外创建。正式测量前先执行 correctness、JIT 和 autotune，避免把首次编译和配置搜索混入稳态延迟。

因此最终 autotuned 数据代表：

```text
当前进程已为这个 M/N 选好配置后的稳态性能
```

它不代表新 shape 第一次调用的 cold-start 延迟。

### 10.4 指标与噪声判断

主要指标：

```text
P50 latency：典型延迟
Core Effective GB/s：2MNs / P50
Speedup：Torch P50 / Triton P50
```

P20/P80 用来判断测量稳定性。如果两个配置的 P50 只差 1%，但分位区间大幅重叠，就只能称为“性能同档”，不能宣布 1% 的稳定领先。

Core Effective GB/s 是统一算法口径，不等于真实 DRAM 带宽；是否发生 register spill、occupancy 下降或 cache miss，需要 profiler 进一步证明。

### 10.5 最终结果文件

| 实验 | CSV | 图片 |
| --- | --- | --- |
| 2 的幂边界 | [`layernorm-boundary.csv`](./benchmark_results/layernorm-boundary.csv) | [`layernorm-boundary.png`](./benchmark_results/layernorm-boundary.png) |
| 固定总元素 | [`layernorm-fixed-numel.csv`](./benchmark_results/layernorm-fixed-numel.csv) | [`layernorm-fixed-numel.png`](./benchmark_results/layernorm-fixed-numel.png) |
| warp sweep | [`layernorm-warp-sweep.csv`](./benchmark_results/layernorm-warp-sweep.csv) | [`layernorm-warp-sweep.png`](./benchmark_results/layernorm-warp-sweep.png) |

---

## 11. 实验一：2 的幂边界如何影响单行 Reduction

### 11.1 实验问题与性能预测

当前 kernel 使用：

```text
BLOCK_SIZE = next_power_of_2(N)
```

因此实验一要验证：当 `N` 只增加 1，却跨过 2 的幂时，性能是否会出现不连续变化。

测试点采用：

```text
127 / 128 / 129
511 / 512 / 513
1023 / 1024 / 1025
2047 / 2048 / 2049
```

控制变量：

```text
固定 M=4096
固定 dtype=FP16
固定输入布局和计时方法
只改变特征维度 N
```

预测是：跨过边界后 `BLOCK_SIZE` 翻倍，padding 比例和 reduction 逻辑宽度突然增加，因此延迟上升、核心 GB/s 下降。

### 11.2 关键结果

| N | BLOCK_SIZE | Auto P50 | Auto GB/s | 相对 PyTorch |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 512 | 0.0274 ms | 306.24 | 2.021x |
| 513 | 1024 | 0.0397 ms | 211.82 | 6.697x |
| 1024 | 1024 | 0.0508 ms | 330.16 | 1.479x |
| 1025 | 2048 | 0.0839 ms | 200.19 | 3.997x |
| 2048 | 2048 | 0.1011 ms | 332.04 | 1.233x |
| 2049 | 4096 | 0.1355 ms | 247.84 | 2.641x |

跨边界变化：

```text
512 -> 513：延迟增加约 45%，GB/s 下降约 31%
1024 -> 1025：延迟增加约 65%，GB/s 下降约 39%
2048 -> 2049：延迟增加约 34%，GB/s 下降约 25%
```

这不是随元素数平滑增长的结果，而是 `BLOCK_SIZE` 离散翻倍形成的结构性台阶。

### 11.3 原因解释

以 `N=1025` 为例：

```text
有效元素：1025
逻辑 BLOCK_SIZE：2048
padding：1023
tile 利用率：约 50%
```

mask 能阻止越界 Global Memory 访问，但无效 lane 仍扩大了后续向量操作和 reduction 的逻辑范围。一个 program 还要维持更宽的 `x`、`centered` 和 affine 中间状态，因此片上资源需求也会增加。

可以由数据支持的因果链是：

```text
N 跨越 2 的幂
-> BLOCK_SIZE 翻倍
-> padding 和 reduction 工作扩大
-> 延迟产生台阶
```

当前不能唯一证明的是：

```text
是否发生 register spill
具体使用了哪种 reduction tree
occupancy 到底下降了多少
```

这些需要 Nsight Compute、PTX/SASS 或编译元数据。

### 11.4 PyTorch 现象与结论边界

PyTorch 在当前环境中对非 2 次幂 shape 也出现更大的性能台阶。例如 `N=511` 明显慢于 `N=512`。分位区间能够复现该现象，但没有分析 PyTorch 内部 dispatch 或生成代码，因此只能称为 shape-sensitive 性能路径。

这也意味着非对齐 shape 上 `5x～8x` 的 Triton speedup 不能泛化为“Triton LayerNorm 普遍快这么多”。其中相当一部分来自当前 PyTorch baseline 在这些 shape 上变慢。

### 11.5 当前最优解

当前单 program-per-row 设计下，最优工程选择是：

1. 保留 mask，保证任意 `N` 的正确性；
2. 用 `M/N` keyed autotune 选择 warp，避免再叠加错误配置造成的损失；
3. 接受 `BLOCK_SIZE` 翻倍带来的剩余台阶；
4. 如果未来必须优化超大且高度非对齐的 `N`，再评估分块 reduction，而不是继续盲调 warp。

autotune 可以减轻配置不合适的问题，但不能消除 `next_power_of_2` 和单 program 整行方案自身的 padding 成本。

### 11.6 面试表达

> 我专门测试了 2 的幂边界，而不是只测整齐 shape。固定 `M=4096`，比较 `512/513`、`1024/1025`、`2048/2049`，发现 `BLOCK_SIZE` 翻倍会带来约 34%～65% 的延迟增长。原因是 padding 和 reduction 逻辑宽度突然扩大，而不是只多计算了一个真实元素。当前用 M/N keyed autotune 避免错误 warp 配置，但 padding 台阶属于单 program 整行算法本身；若要继续解决，需要考虑分块 reduction，并用 profiler 验证资源瓶颈。

---

## 12. 实验二：固定总元素数，分离总工作量与 Shape 效应

### 12.1 实验问题与控制变量

只比较不同 shape 时，较大矩阵自然更慢，无法判断是总工作量还是形状造成的。因此实验二固定：

```text
M * N = 4,194,304
```

然后逐步减少行数、增加每行特征维度：

```text
16384 x 256
8192  x 512
...
128   x 32768
```

控制变量包括总元素数、dtype、输入布局、计时方法和候选 warp 集合；主要变化只有 `M/N` 比例。

要验证的核心问题是：相同 `M*N` 是否意味着相同性能。

### 12.2 关键结果

| M | N | Auto P50 | Auto GB/s | 相对 PyTorch |
| ---: | ---: | ---: | ---: | ---: |
| 16384 | 256 | 0.0569 ms | 294.88 | 2.175x |
| 8192 | 512 | 0.0552 ms | 303.94 | 2.343x |
| 4096 | 1024 | 0.0540 ms | 310.60 | 1.737x |
| 2048 | 2048 | 0.0551 ms | 304.64 | 1.162x |
| 1024 | 4096 | 0.0520 ms | 322.64 | 1.345x |
| 512 | 8192 | 0.0529 ms | 316.98 | 1.239x |
| 256 | 16384 | 0.0532 ms | 315.08 | 1.410x |
| 128 | 32768 | 0.0652 ms | 257.51 | 1.525x |

### 12.3 现象分区

结果不是“`N` 越长越慢”，而是分为两个区域。

稳定平台：

```text
N=256～16384
P50 约 0.052～0.057 ms
Core GB/s 约 295～323
```

在这个范围内，行数从 16384 降到 256，autotuned kernel 仍能维持接近的稳态吞吐。说明这些 shape 中仍有足够 program，并且单 program reduction 资源尚未造成断崖式退化。

极长 reduction：

```text
M=128, N=32768
P50=0.0652 ms
Core GB/s=257.51
```

相对平台峰值约下降 20%。这证明总元素相同并不保证相同性能，shape 会改变 program 数、每个 program 的工作宽度和资源配置。

### 12.4 为什么不能只归因于并行 program 不足

`N=32768` 时：

```text
program 数 = M = 128
GPU SM 数 = 26
平均仍约有 5 个 program/SM
```

所以不能简单说“只有几个 program，绝大多数 SM 空闲”。更完整的解释是：

```text
M 减少，能够调度的独立 program 变少
+ 每个 program 的 reduction 宽度大幅增加
+ 单 program 的片上状态和执行映射压力增大
-> 延迟隐藏和吞吐共同受到影响
```

warp sweep 进一步证明 `N=32768` 的问题不仅是 grid 大小：在完全相同的 128 个 program 下，仅把 `w4` 改为 `w8`，延迟就能显著下降。

### 12.5 当前最优解

当前测试范围内：

- `N=256～16384` 已形成约 `300+ GB/s` 平台，不需要为每个点编写不同算法；
- `N=32768` 需要更多协作 warp，autotune 能恢复到约 `257.5 GB/s`；
- 如果 `N` 继续增大，当前单 program 整行实现最终会碰到资源上限，应转向分块 reduction。

因此当前最优策略不是固定一个 warp，而是先保持同一数学 kernel，再让 autotune 根据 `M/N` 选择执行配置。

### 12.6 面试表达

> 为了区分总数据量和 shape 的影响，我固定 `M*N=2^22`，只改变行数和特征维度。结果显示 `N=256～16384` 都能维持约 295～323 GB/s，但 `N=32768` 降到约 258 GB/s。这说明 LayerNorm 性能不只取决于总元素数，还取决于 program 数和单行 reduction 宽度。由于此时仍有 128 个 program、GPU 只有 26 个 SM，所以我没有把下降简单归因于并行度不足，而是结合 warp sweep 判断单 program 执行映射和资源压力也很关键。

---

## 13. 实验三：用单变量 Sweep 选择 num_warps

### 13.1 为什么单独测试 num_warps

`num_warps` 改变一个 Triton program 使用多少协作 warp，但不会改变：

```text
数学结果
program 总数
每行真实元素数量
Global Memory 中的输入输出布局
```

因此固定 shape，只改变 `w1/w2/w4/w8`，可以较干净地观察 program 内并行配置的影响。

为避免 baseline 漂移，每个 shape 只创建一次输入、只测一次 PyTorch，然后让四个 Triton 配置共享同一输入和参考。

### 13.2 关键结果

| M,N | w1 P50 | w2 P50 | w4 P50 | w8 P50 | 当前判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| 8192,512 | 0.0545 | 0.0573 | 0.0602 | 0.0928 | w1/w2 较好，w8 明显过重 |
| 2048,2048 | 0.0512 | 0.0526 | 0.0530 | 0.0532 | 四组分位区间重叠，性能同档 |
| 512,8192 | 0.1024 | 0.0548 | 0.0520 | 0.0530 | w1 明显不足，w2/w4/w8 同档 |
| 128,32768 | 0.3392 | 0.3272 | 0.1115 | 0.0581 | w8 显著最佳 |

### 13.3 逐区间解释

小 reduction，`N=512`：

```text
w1 = 0.0545 ms
w8 = 0.0928 ms
```

此时工作量不足以抵消更多协作 warp 带来的调度、通信和资源开销。数据只证明 `w8` 明显更差，不能在没有底层代码的情况下断言一定由 shared-memory barrier 导致。

中等 reduction，`N=2048`：

四个 P50 只差约 4%，且 P20/P80 区间重叠，说明已经进入对 warp 数不敏感的平台。这里不能因为 `w1` 的 P50 最小就宣布它稳定最优。

较长 reduction，`N=8192`：

`w1` 延迟约为其他配置的两倍，说明单个 warp 无法高效承接超宽 reduction；但 `w2/w4/w8` 的差距很小，应视为同档。

极长 reduction，`N=32768`：

```text
w1 = 0.3392 ms
w2 = 0.3272 ms
w4 = 0.1115 ms
w8 = 0.0581 ms
```

`w8` 相比 `w4` 快约 `1.92x`，相比 `w1` 快约 `5.84x`，分位区间也明显分离。这是当前最强的配置相关证据：极宽 reduction 必须使用足够多的协作 warp。

### 13.4 参数规律与证据边界

数据支持的总体趋势是：

```text
短行：warp 太多会浪费资源
中等行：多个配置可能同档
长行：warp 太少开始明显变慢
极长行：w8 显著最佳
```

但最佳值不是一个已经被证明的单调函数。`num_warps` 最终影响哪些寄存器、shared memory 和 occupancy 指标，仍需 profiler 证明。

### 13.5 当前最优解

如果只为当前四个代表 shape 手写选择，可以得到：

```text
N=512：w1/w2
N=2048：任一候选基本同档
N=8192：w2/w4/w8
N=32768：w8
```

但工程代码不能只覆盖四个点，也不能把噪声中的微小领先硬编码成规则，因此最终选择保留 manual 入口，并增加 autotune 处理未测试的中间 shape。

### 13.6 面试表达

> 我没有直接假设 num_warps 越大越快，而是在固定总元素和 shape 后只 sweep w1/w2/w4/w8。结果表明 N=512 时 w8 明显更慢，N=2048 时四组同档，N=8192 时 w1 明显不足，而 N=32768 时 w8 比 w4 快约 1.9 倍。这个实验说明最佳 warp 数与 reduction 宽度强相关，但不是一个简单、已证明的单调函数，所以我保留手动入口做因果实验，生产入口则使用 M/N keyed autotune。

---

## 14. 失败优化实验：从单调 Heuristic 到 Autotune

### 14.1 为什么最初会提出 Heuristic

根据 warp sweep 的两端现象，曾提出：

```python
num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
```

它的直觉是：小 tile 用少量 warp，宽 reduction 随 `BLOCK_SIZE` 增大而增加 warp，并把上限限制为 8。

这个规则不是拍脑袋产生的：`N=512` 不需要 w8，而 `N=32768` 强烈需要 w8，所以单调增加看起来合理。

### 14.2 数据如何推翻它

实际三方实验发现：

```text
N=1025, BLOCK=2048
manual w4：约 0.0730 ms
heuristic w8：约 0.2706 ms
heuristic 慢约 3.7x

N=2049, BLOCK=4096
manual w4：约 0.1429 ms
heuristic w8：约 0.3700 ms
heuristic 慢约 2.6x
```

但同一个 heuristic 在 `N=32768` 上选择 `w8` 又是正确方向。

因此失败的不是“大 reduction 可能需要更多 warp”这个局部观察，而是把它推广成只依赖 `BLOCK_SIZE` 的全局单调规则。

### 14.3 失败原因

只看 `BLOCK_SIZE` 会丢失：

- 真实 `N` 和 tile 利用率；
- padding 比例；
- 行数 `M` 和全局 program 数；
- 编译器针对不同 shape 生成的 reduction layout；
- warp 配置与片上资源的非线性交互。

例如 `N=1025` 和 `N=2048` 都使用 `BLOCK_SIZE=2048`，但前者约一半 lane 是 padding。只依据 `BLOCK_SIZE` 无法区分它们。

### 14.4 最终方案

autotune 候选：

```text
w1 / w2 / w4 / w8
```

autotune key：

```text
M / N
```

最终保留三类入口：

```text
layer_norm(..., num_warps=...)
    -> manual，供控制变量和跨硬件调试

layer_norm_heuristic(...)
    -> 保留失败实验，不能作为最终 optimized 入口

layer_norm_autotuned(...)
    -> 对新 M/N 实测候选，缓存后用于稳态执行
```

autotune 也不是没有代价：新 shape 第一次执行需要测量候选；微秒级候选还可能受噪声影响。因此它解决的是“避免灾难性错误配置和处理 shape 差异”，而不是保证每次 P50 都严格超过 manual。

### 14.5 这次失败实验的价值

这次最重要的优化经验不是“最后用了 autotune”，而是形成了可验证的决策流程：

```text
从硬件直觉提出假设
-> 设计单变量实验
-> 找到能支持和反驳假设的 shape
-> 不掩盖失败结果
-> 修正性能模型
-> 再选择更稳健的工程方案
```

这比只保留一个最快结果更能说明具备算子性能分析能力。

---

## 15. 完整优化决策链与面试表达

### 15.1 从正确实现到最终配置

本阶段的完整优化链路是：

```text
1. 先实现一行一个 program 的正确 LayerNorm
2. 用 FP32 完成 mean/variance reduction
3. 融合 normalize、weight 和 bias，避免全局中间张量
4. 用固定 w4 建立 manual baseline
5. 测 2 的幂边界，发现 BLOCK_SIZE/padding 性能台阶
6. 固定总元素数改变 M/N，发现中等区间平台和极长行退化
7. 单变量 sweep num_warps，确认不同 N 的最佳配置不同
8. 提出 BLOCK_SIZE 单调 heuristic
9. 用反例发现 heuristic 在部分非对齐 shape 上严重退化
10. 改用 M/N keyed autotune，并保留 manual 入口
11. 用 PyTorch/manual/autotuned 三方 Benchmark 验证最终结果
```

这条链路体现的是“先建立性能模型，再用证据修正模型”，而不是直接堆 autotune 配置。

### 15.2 最终结果如何表述

最终两组主实验中，autotuned Triton 在本轮所有测试 shape 上都快于 PyTorch baseline；中等特征维度的 Core Effective GB/s 主要处于约 `300～323 GB/s`。

可靠结论：

> 当前 Triton LayerNorm 使用 row-wise FP32 reduction 和 elementwise fusion，不物化 centered、squared、normalized 等全局中间张量；边界和 warp sweep 证明性能对 N、padding 和 program 内并行配置敏感，M/N keyed autotune 能避免严重错误的 warp 配置，并在当前 GPU 的测试范围内获得有竞争力的稳态性能。

必须同时说明的限制：

- 数据来自 RTX 5060 Laptop、FP16 和有限 shape；
- PyTorch 本身通常也是 fused LayerNorm；
- 非 2 次幂上的巨大 speedup 与 PyTorch 当前 shape-sensitive 路径有关；
- autotune 结果不包含首次选型成本；
- Effective GB/s 不是物理 DRAM 带宽；
- 没有 profiler 证据证明 register spill、occupancy 或具体 reduction lowering。

### 15.3 面试问题：你怎么测试算子性能？

推荐回答：

> 我不会只测一个整齐的大 shape。首先明确计时范围，我测的是包含 wrapper、输出分配、launch 和 kernel 的算子级延迟；输入分配、JIT 和 autotune 放在正式计时之外。计时采用 100 ms warmup、300 ms 重复测量，并记录 P20/P50/P80，主要比较 P50，同时用分位区间判断差异是否超过噪声。
>
> 指标除了延迟，还按主输入一次读、输出一次写计算 2MNs 的 Core Effective GB/s，并明确它不等于真实 DRAM 带宽。shape 方面，我分别测试 2 的幂边界、固定总元素改变 M/N，以及固定 shape sweep num_warps，从而把 padding、shape 和参数配置的影响拆开。

### 15.4 面试问题：你怎么优化参数？

推荐回答：

> 我先用固定 w4 建立 baseline，然后一次只改 num_warps，比较 w1/w2/w4/w8。
>
> 实验发现 N=512 时 w8 明显更慢，N=2048 时配置基本同档，N=8192 时 w1 明显不足，N=32768 时 w8 比 w4 快约 1.9 倍。因此不存在统一最优 warp。
>
> 我曾根据 BLOCK_SIZE 写过单调 heuristic，但 N=1025 和 N=2049 等反例让它出现 2～4 倍退化，说明只看 BLOCK_SIZE 会忽略真实 N、padding 和 M。最终我保留 manual 入口用于控制变量实验，并使用 M/N keyed autotune 在四个合理候选中选择配置。

### 15.5 面试问题：你的算子为什么更快、是否更省显存？

推荐回答：

> kernel 在一个 program 内完成 mean、variance、normalize 和 affine，主输入源码级读取一次、输出写回一次，不把 centered、squared 和 normalized 写回 Global Memory，因此减少了中间显存流量。
>
> 不过 PyTorch LayerNorm 本身通常也是 fused 实现，所以我不会宣称相对 PyTorch 一定大幅节省显存容量；当前收益主要来自 shape 和 warp 配置。是否发生物理 spill、真实 DRAM 流量是多少，还需要 profiler 证明。

### 15.6 面试问题：为什么相信这些性能结论？

推荐回答：

> 我用正确性测试先建立底线，最终是 30 passed；
>
> Benchmark 中把 JIT 和 autotune 移出计时，使用多个分位数而不是单次最小值；优化实验坚持控制变量，并同时保留失败 heuristic 的反例。对于分位区间重叠的微小差异，我只称为同档；对于 register spill、occupancy 等没有 profiler 的部分，我只提出假设，不做唯一归因。

---

## 16. 当前限制与延期项

当前未完成：

- backward；
- Welford；
- profiler 对 register、spill、occupancy、L2/DRAM throughput 的验证；
- 大于单 program 上限后的分块 reduction；
- residual + LayerNorm、bias + LayerNorm、LayerNorm + activation 等进一步融合；
- 任意维度、布局和混合精度工程支持；
- 训练场景中 mean/rstd 的保存与重计算权衡。

这些内容不阻塞既定阶段三路线。

---

## 17. 本阶段形成的性能观

1. Reduction 的 mask 不仅要防越界，还必须保证 padding 不改变数学结果。
2. 中心化方差比 `E[x^2]-E[x]^2` 更稳定，但 FP32 reduction 仍不是无限精度。
3. Fusion 的核心价值是缩短中间张量生命周期、减少 Global Memory traffic。
4. 一次源码 load/store 不等于物理上绝对无 spill。
5. `BLOCK_SIZE` 跨越 2 的幂会带来 padding 和资源台阶。
6. 固定总元素数仍不足以唯一决定性能，`M/N` 的形状同样重要。
7. `num_warps` 不是越大越好，最佳值随 reduction 宽度和执行映射变化。
8. 简单启发式可能在部分 shape 上严重失败，必须用受控 Benchmark 验证。
9. Autotune 用于处理参数交互和 shape 差异，但必须区分 cold-start 与 steady-state。
10. 没有 profiler 证据时，对 register spill、shared-memory 路径和 occupancy 只提出假设，不做唯一归因。

---

## 18. 与后续路线的连接

`04_layernorm` 已完成：

- 数学定义与手算；
- Triton row-wise fused forward；
- 均值/方差 reduction；
- FP32 中间计算与数值稳定压力测试；
- elementwise fusion 与一次主输入读写分析；
- 普通、边界、dtype、错误输入测试；
- 2 的幂边界、固定总元素和 warp sweep；
- heuristic 失败实验；
- M/N keyed autotune；
- CSV、图片和复盘笔记。

后续严格按既定顺序：

```text
05_attention_naive
-> 显式 QK^T
-> scale / mask / row-wise softmax
-> P @ V
-> 中间 score/probability 矩阵生命周期
-> correctness baseline
-> Benchmark

06_flash_attention_forward
-> tiled QK^T
-> online softmax
-> 不物化完整注意力矩阵
```

`05_attention_naive` 的任务是建立可信的 Attention 正确性和显存代价基线，不提前实现 FlashAttention

## kernel 代码

```py
import torch
import triton
import triton.language as tl


SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
MAX_FUSED_SIZE = 65536

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    x_row_stride,
    y_row_stride,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # 一个 program 负责一整行，行与行之间没有 reduction 依赖。
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x_row_ptr = x_ptr + row_idx * x_row_stride
    x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)

    # FP16/BF16 的表示范围和有效位不足以稳定承担 reduction，先转成 FP32。
    x = x.to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols

    # padding 位置虽然加载为 0，但减去 mean 后会变成 -mean；必须再次清零，
    # 否则无效位置的 mean^2 会进入方差 reduction。
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    rstd = tl.rsqrt(variance + eps)

    # 将归一化、缩放和平移融合在同一个 kernel 中，不生成全局中间张量。
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = centered * rstd * weight + bias

    y_row_ptr = y_ptr + row_idx * y_row_stride
    tl.store(y_row_ptr + offsets, y, mask=mask)


AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=1),
    triton.Config({}, num_warps=2),
    triton.Config({}, num_warps=4),
    triton.Config({}, num_warps=8),
]

# M/N 都进入 key：特征维度决定单行 reduction，行数会改变全局并行度。
autotuned_layer_norm_kernel = triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["n_rows", "n_cols"],
)(layer_norm_kernel)

# 条件判断
def _validate_layer_norm_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[int, int]:
    """统一 manual、heuristic 和 autotuned launcher 的输入契约。"""
    if not isinstance(x, torch.Tensor):
        raise TypeError("x 必须是 torch.Tensor")
    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight 必须是 torch.Tensor")
    if not isinstance(bias, torch.Tensor):
        raise TypeError("bias 必须是 torch.Tensor")
    if not x.is_cuda or not weight.is_cuda or not bias.is_cuda:
        raise ValueError("x、weight 和 bias 必须位于 CUDA 设备")
    if x.ndim != 2:
        raise ValueError("x 必须是二维矩阵，LayerNorm 沿最后一维计算")
    if weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("weight 和 bias 必须是一维张量")
    if x.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"不支持 dtype={x.dtype}，仅支持 FP16、BF16 和 FP32")
    if weight.dtype != x.dtype or bias.dtype != x.dtype:
        raise TypeError("x、weight 和 bias 的 dtype 必须一致")
    if weight.device != x.device or bias.device != x.device:
        raise ValueError("x、weight 和 bias 必须位于同一 CUDA 设备")
    if x.stride(1) != 1:
        raise ValueError("x 的最后一维必须连续，因为 kernel 使用 offsets 直接寻址")
    if not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("weight 和 bias 必须连续")
    if not isinstance(eps, (int, float)) or eps < 0:
        raise ValueError("eps 必须是非负实数")

    n_rows, n_cols = x.shape
    if weight.numel() != n_cols or bias.numel() != n_cols:
        raise ValueError("weight 和 bias 的长度必须等于 x 的最后一维")
    return n_rows, n_cols

# 根据特征维度选择合适的 block size
def _block_size_for(n_cols: int) -> int:
    block_size = triton.next_power_of_2(n_cols) if n_cols > 0 else 1
    if block_size > MAX_FUSED_SIZE:
        raise ValueError(
            f"特征维度 {n_cols} 超过单个 program 的支持范围；"
            "后续需要改用分块 reduction"
        )
    return block_size


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    num_warps: int = 4,
) -> torch.Tensor:
    # 对二维 CUDA Tensor 的最后一维执行 fused LayerNorm forward。
    n_rows, n_cols = _validate_layer_norm_inputs(x, weight, bias, eps)
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps 必须是 1、2、4 或 8")

    y = torch.empty_like(x)
    if x.numel() == 0:
        return y

    block_size = _block_size_for(n_cols)

    # 当前先固定一行一个 program；后续通过受控实验研究 num_warps 和特征维度。
    grid = (n_rows,)
    layer_norm_kernel[grid](
        x,
        weight,
        bias,
        y,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return y

# 根据 reduction 逻辑宽度选择 program 内的协作 warp 数
def select_num_warps(block_size: int) -> int:
    if block_size <= 0:
        raise ValueError("block_size 必须为正整数")
    return min(max(block_size // 256, 1), 8)

# 根据启发式选择合适的 block size 和 num_warps
def layer_norm_heuristic(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    # 保留已验证并非全局可靠的 BLOCK_SIZE 单调启发式，供复现实验。
    _, n_cols = _validate_layer_norm_inputs(x, weight, bias, eps)
    block_size = _block_size_for(n_cols)
    num_warps = select_num_warps(block_size)

    return layer_norm(
        x,
        weight,
        bias,
        eps=eps,
        num_warps=num_warps,
    )

# 自动调优的 LayerNorm
def layer_norm_autotuned(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    # 在 w1/w2/w4/w8 中按 M/N 实测选择配置。
    n_rows, n_cols = _validate_layer_norm_inputs(x, weight, bias, eps)

    y = torch.empty_like(x)
    if x.numel() == 0:
        return y

    block_size = _block_size_for(n_cols)
    grid = (n_rows,)
    autotuned_layer_norm_kernel[grid](
        x,
        weight,
        bias,
        y,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
    )
    return y
```
