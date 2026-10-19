# 06 FlashAttention Forward 学习笔记

## 1. 本阶段完成了什么

本阶段实现了一个容易阅读、可以验证和调优的 Triton FlashAttention Forward：

- 输入 `Q/K/V` 的形状为 `[B, H, N, D]`；
- 支持非因果和因果 Attention；
- 支持 FP16、BF16 和 FP32；
- 支持非对齐的 `N`、`D`，当前要求 `1 <= D <= 128`；
- 在一个 Kernel 中完成 `QK^T`、缩放、遮罩、Softmax 和 `P@V`；
- 只返回输出 `O`，不把完整的 Score `S` 和 Probability `P` 写回显存；
- 最终统一配置为 `BLOCK_M=64`、`BLOCK_N=32`、`num_warps=4`；
- 因果模式下，`N>=512` 时跳过完全位于未来的 Key Tile。

本阶段只实现 forward，不包含 backward、训练反向传播、Dropout，也不追求 FlashAttention 2/3 的复杂优化。

相关文件：

| 文件 | 作用 |
|---|---|
| `kernel.py` | FlashAttention Forward Kernel、输入验证和调优入口 |
| `test_correctness.py` | correctness、边界和错误输入测试 |
| `benchmark.py` | PyTorch Naive、Triton Naive、Flash 的正式对比 |
| `benchmark_tuning.py` | 第一轮 Tile/warp 控制变量实验 |
| `benchmark_final_selection.py` | 候选配置的三轮随机顺序复测 |
| `benchmark_causal_skip.py` | 因果 Key Tile 跳块实验 |
| `benchmark_results/` | CSV、延迟图、显存图和调优图 |

## 2. 从 Naive Attention 到 FlashAttention

### 2.1 数学定义

对每个 batch 和 head：

\[
S=\frac{QK^T}{\sqrt D},\qquad
P=\operatorname{softmax}(S),\qquad
O=PV
\]

其中：

- `Q/K/V` 的形状都是 `[N, D]`；
- `S/P` 的形状都是 `[N, N]`；
- `O` 的形状是 `[N, D]`。

这里的 `N` 是序列长度，`D` 是每个 token 在一个 head 中的特征数。

### 2.2 Naive 为什么必须保存 S 和 P

05 的 Naive 实现把计算拆成三个 Kernel：

```text
Score Kernel       写出完整 S
Softmax Kernel     读 S，写出完整 P
Output Kernel      读 P 和 V，写 O
```

后一个 Kernel 无法直接使用前一个 Kernel 的片上临时数据，所以中间结果只能先写到全局显存（Global Memory）。因此 Naive 实现需要两个 `[B,H,N,N]` 张量。

以 `B=2、H=8、N=1024、FP16` 为例：

\[
2\times8\times1024\times1024\times2\text{ bytes}=32\text{ MiB}
\]

单个 `S` 占 32 MiB，`S+P` 占 64 MiB。如果只计算两者各写一次、读一次，中间张量也至少产生约 128 MiB 的显存流量。

当 `N` 翻倍时，`S/P` 的容量变成四倍。这就是 `N^2` 中间矩阵的核心问题。

### 2.3 FlashAttention 改变了什么

FlashAttention 不改变 Attention 的数学结果，而是改变中间数据的保存方式：

```text
加载一个 Q Tile
    -> 依次加载多个 K/V Tile
    -> 在片上计算当前 Score Tile
    -> 在线更新 Softmax 和输出累加器
    -> 当前 Tile 用完即丢弃
最终只把 O 写回全局显存
```

因此它减少的是：

- `N×N` 中间张量的显存容量；
- S/P 写回和再次读取造成的显存流量；
- 多个独立 Kernel 之间的中间结果传递。

它没有把计算复杂度从 `O(N^2)` 变成 `O(N)`。`QK^T` 和 `P@V` 仍然需要处理约 `N^2D` 的乘加。

## 3. Program、Tile 和数据范围

### 3.1 为什么同时有 M 和 N

Score 矩阵的行来自 Query，列来自 Key：

- `M` 表示一次处理多少个 Query 行；
- `N` 表示一次处理多少个 Key 列。

因此 `BLOCK_M` 和 `BLOCK_N` 可以不同。当前正式配置中：

```text
Q Tile   [64, D]
K^T Tile [D, 32]
V Tile   [32, D]
Score    [64, 32]
acc      [64, D]
m / l    [64]
```

### 3.2 Program 映射

Grid 为：

\[
(\lceil N/BLOCK\_M\rceil,\ B\times H)
\]

第一个 Program 维度选择 Query Tile，第二个维度展开 batch 和 head：

```text
batch_idx = batch_head_idx // H
head_idx  = batch_head_idx % H
```

一个 Program 固定处理：

- 一个 batch；
- 一个 head；
- 一组连续的 Query 行；
- 内部循环遍历该组 Query 需要的全部 Key/Value Tile；
- 最终写出这些 Query 行对应的 `O`。

Key Tile 不能简单拆给互不通信的 Program，因为同一个 Query 行跨 Key Tile 的运行最大值 `m`、分母 `l` 和累加器 `acc` 必须连续更新。拆开后还需要额外的跨 Program 合并与同步。

## 4. 为什么局部 Softmax 不能直接相加

假设一行 Score 分成两个 Key 块。分别做普通 Softmax 后，每块概率之和都是 1，两块直接拼起来后总和变成 2，不再是整行的正确概率。

根本原因是：Softmax 的分母必须包含整行所有 Key：

\[
p_j=\frac{e^{s_j}}{\sum_{t=1}^{N}e^{s_t}}
\]

处理第一个块时还不知道后续块的 Score，所以必须保存能够继续合并的统计量，而不是过早得到一个不可合并的局部概率。

## 5. 在线 Softmax 推导

### 5.1 三个状态分别表示什么

对一个 Query 行，处理到当前 Key 块时保存：

\[
m=\max(s_1,\ldots,s_t)
\]

`m` 是目前见过的运行最大值（running maximum）。

\[
l=\sum_{j=1}^{t}e^{s_j-m}
\]

`l` 是以当前 `m` 为基准的未归一化权重总和。

\[
acc=\sum_{j=1}^{t}e^{s_j-m}V_j
\]

`acc` 是以同一个 `m` 为基准的未归一化 Value 加权和。

最终：

\[
O=\frac{acc}{l}
\]

因为分子和分母使用相同的指数基准，除法后正好得到 `softmax(S)@V`。

### 5.2 合并一个新 Key 块

旧状态为 `m_old、l_old、acc_old`，新块 Score 为 `s`。

第一步，求统一的新最大值：

\[
m_{new}=\max(m_{old},\max(s))
\]

第二步，把旧状态换到新基准：

\[
\alpha=e^{m_{old}-m_{new}}
\]

旧权重原来按 `m_old` 计算，现在必须乘 `alpha`：

\[
e^{s_j-m_{old}}\alpha=e^{s_j-m_{new}}
\]

第三步，计算当前块相对于 `m_new` 的权重：

\[
w=e^{s-m_{new}}
\]

第四步，同时更新分母和输出累加器：

\[
l_{new}=\alpha l_{old}+\sum w
\]

\[
acc_{new}=\alpha acc_{old}+wV
\]

当新块出现更大最大值时，如果只缩放 `l_old` 而不缩放 `acc_old`，旧 Value 的实际占比仍然会偏大。因此二者必须乘同一个 `alpha`。

### 5.3 一个完整手算

设一行 Score 分成两块：

```text
第一块 Score = [0, 0]，Value = [1, 3]
第二块 Score = [1, 1]，Value = [5, 7]
```

这里为方便手算，每个 Value 只有一个数。

第一块处理后：

\[
m_{old}=0
\]

\[
l_{old}=e^0+e^0=2
\]

\[
acc_{old}=1\times1+1\times3=4
\]

第二块带来了更大最大值：

\[
m_{new}=1,\qquad \alpha=e^{0-1}=e^{-1}
\]

所以：

\[
l_{new}=2e^{-1}+2
\]

\[
acc_{new}=4e^{-1}+5+7
\]

最终：

\[
O=\frac{4e^{-1}+12}{2e^{-1}+2}\approx4.92
\]

直接对 `[0,0,1,1]` 做完整 Softmax 也会得到同一结果。整个过程中不需要保存完整概率数组。

## 6. 一个 Kernel 内如何衔接完整 Attention

每个 Program 的主流程可以概括为：

```text
加载 Q Tile 一次
初始化 m=-inf、l=0、acc=0

遍历 Key/Value Tile：
    加载 K^T Tile 和 V Tile
    scores = Q Tile @ K^T Tile
    scores *= 1/sqrt(D)
    应用越界与因果遮罩
    在线更新 m、l、acc

O Tile = acc / l
只写回有效的 Query 行和特征
```

Q Tile 会跨全部 Key Tile 复用；每个 K/V Tile 只在当前循环使用。Score Tile 只在片上短暂存在，不会写回 Global Memory。

## 7. 因果遮罩和非对齐边界

### 7.1 因果语义

Query 位置为 `i`，Key 位置为 `j`。因果 Attention 只允许：

\[
j\le i
\]

分块后，一个 Key Tile 对当前 Query Tile 可能是：

- 完全可见：所有位置都可能被使用；
- 部分可见：Tile 内仍需逐元素判断 `j<=i`；
- 完全位于未来：整个 Tile 可以跳过。

正式入口只在 `causal=True` 且 `N>=512` 时启用完全未来 Tile 跳过。这个阈值来自当前设备实验，不是数学定理。

### 7.2 非对齐 N 和 D

最后一个 Tile 可能超出真实形状，因此分别建立：

- `query_is_valid`：Query 行是否小于 `N`；
- `key_is_valid`：Key 行是否小于 `N`；
- `feature_is_valid`：特征位置是否小于 `D`。

无效 Q/K/V 位置不参与有效计算；无效 Score 被视为负无穷，随后权重显式置零；最终只写回有效输出位置。

`BLOCK_D` 补齐到不小于 16 的二次幂，例如 `D=40` 使用 `BLOCK_D=64`，多出的特征由 mask 保护。

## 8. 数值精度

在线 Softmax 的 `m、l、acc` 使用 FP32，原因是这些状态会跨多个 Key Tile 累积，低精度更容易产生误差。

当前实现的精度策略：

- 输入可以是 FP16、BF16 或 FP32；
- Score 完成缩放后参与稳定 Softmax；
- `m、l、acc` 用 FP32 保存；
- FP16/BF16 的块权重转换到输入 dtype 后进入 `P@V` 的 `tl.dot`；
- FP32 输入使用 IEEE 精度路径；
- 最终 `acc/l` 写回输出，自动转换为输入 dtype。

减去最大值不是为了改变概率，而是避免 `exp(很大的数)` 溢出。最大 Score 对应的指数始终为 1，其余指数位于 `(0,1]`。

## 9. Global Memory、片上存储和压力

### 9.1 中间张量生命周期

| 数据 | 位置与生命周期 |
|---|---|
| Q/K/V | 已存在的 Global Memory 输入 |
| Q Tile | Program 内加载一次，跨 Key 循环复用 |
| K/V Tile | 每轮循环加载，用完即丢弃 |
| Score/权重 Tile | 当前循环片上临时数据，不写回显存 |
| m/l/acc | Program 内跨 Key 循环保存 |
| O | 最终写回 Global Memory |

这里把“片上”作为便于理解的总称。具体变量最终放在寄存器还是其他片上资源、是否发生寄存器溢出，需要编译信息和 Profiler 才能确认。

### 9.2 Tile 不是越大越好

更大的 Tile 可以：

- 减少 Program 数和循环次数；
- 增加 Q/K/V 的数据复用；
- 提高单次矩阵乘规模。

但也会放大 Score Tile 和 `acc[BLOCK_M,D]`，增加寄存器压力，可能降低同时驻留的 Program 数。`num_warps` 同样存在并行度与资源占用的权衡，所以必须用实验选择，而不能只看形状大小。

## 10. Correctness 验证

本阶段测试覆盖：

- 对齐和非对齐 `N/D`；
- 极小序列、空序列；
- 不同 batch/head；
- FP16、BF16、FP32；
- 因果和非因果；
- 多个 Key Tile 的在线合并；
- 与 05 PyTorch 显式 reference 对比；
- 与 05 Triton Naive 交叉验证；
- 所有调优候选配置；
- 因果跳块的极小、非对齐和多 batch/head Shape；
- 非 Tensor、CPU Tensor、维度/Shape/dtype/device/连续性错误；
- 非法 `D`、Tile、warp 和布尔参数；
- 正式入口的 `N=512` 因果跳块阈值。

本轮结果为 77 项通过；不支持 BF16 的设备预期为 69 项通过、8 项跳过。

正式 Benchmark 在每个 Shape 计时前还会再次把 Triton Naive 和 Flash 输出与 PyTorch reference 比较。因此正式 CSV 能完整生成，也验证了 `N>=512` 的自动跳块路径。

## 11. Benchmark 环境和口径

### 11.1 环境与 Shape

沿用本项目环境：

```text
GPU：NVIDIA GeForce RTX 5060 Laptop GPU
PyTorch：2.11.0+cu128
Triton：3.6.0
CUDA：12.8
环境：WSL / Laptop GPU
```

正式 Benchmark 固定：

```text
B=2, H=8, D=64, dtype=FP16
N=128, 256, 512, 1024, 2048
causal=False / True
warmup=100 ms, rep=300 ms
P20/P50/P80=(0.2, 0.5, 0.8) 分位数
```

三个比较对象：

- PyTorch Naive：显式 `matmul -> mask/scale -> softmax -> matmul`；
- Triton Naive：05 的三个 Kernel，显式保存 S/P；
- Triton Flash：06 的融合 Kernel，不保存完整 S/P。

PyTorch SDPA 不在本轮对比中，因为它可能自动选择融合后端，不再是显式 Naive 算法。

Q/K/V 在计时前创建，correctness、JIT 和预热也在正式计时前完成。因此延迟是已编译后的稳态算子时间，不包含首次编译。

### 11.2 TFLOPS 口径

两个矩阵乘的完整等效计算量约为：

\[
FLOPs=4BHN^2D
\]

\[
TFLOPS=\frac{FLOPs}{latency_{ms}}\times10^{-9}
\]

因果跳块后，CSV 仍使用完整 `N^2` 计算量，因此表示“完整问题等效吞吐”，不是实际执行指令吞吐。实际 Tensor Core 利用率需要 Profiler。

### 11.3 峰值显存口径

测量流程为：

1. Q/K/V 已经存在；
2. 重置 CUDA 峰值统计；
3. 记录调用前后的活跃显存峰值增量。

所以结果主要反映 S/P/O 和临时张量，不包含输入 Q/K/V，也不等于 CUDA 缓存分配器保留的全部显存。

## 12. 正式 Benchmark 结果

### 12.1 P50 延迟和加速比

| 模式 | N | PyTorch ms | Triton Naive ms | Flash ms | 对 PyTorch | 对 Naive |
|---|---:|---:|---:|---:|---:|---:|
| 非因果 | 128 | 0.0315 | 0.0313 | 0.0169 | 1.86x | 1.85x |
| 非因果 | 256 | 0.0684 | 0.0542 | 0.0574 | 1.19x | 0.94x |
| 非因果 | 512 | 0.1620 | 0.1380 | 0.1312 | 1.23x | 1.05x |
| 非因果 | 1024 | 0.7938 | 0.5771 | 0.3171 | 2.50x | 1.82x |
| 非因果 | 2048 | 3.6451 | 3.5750 | 0.7399 | 4.93x | 4.83x |
| 因果 | 128 | 0.0451 | 0.0333 | 0.0329 | 1.37x | 1.01x |
| 因果 | 256 | 0.0790 | 0.0500 | 0.0449 | 1.76x | 1.11x |
| 因果 | 512 | 0.2566 | 0.1382 | 0.1103 | 2.33x | 1.25x |
| 因果 | 1024 | 1.3715 | 0.8397 | 0.1472 | 9.32x | 5.70x |
| 因果 | 2048 | 6.1057 | 3.1707 | 0.4207 | 14.51x | 7.54x |

结论：

- 小序列主要受固定开销和测量波动影响，Flash 不保证每个点都更快；
- `N=1024/2048` 时，避免 S/P 全局显存往返的优势已经非常明显；
- 因果长序列还叠加了未来 Key Tile 跳过，因此收益大于非因果；
- `N=256/512` 的 P20～P80 波动较宽，小差距不应形成强结论。

### 12.2 P20/P50/P80 和 TFLOPS

代表性 Flash 结果：

| 模式 | N | P20/P50/P80 ms | 完整等效 TFLOPS |
|---|---:|---:|---:|
| 非因果 | 1024 | 0.2514 / 0.3171 / 0.3315 | 13.55 |
| 非因果 | 2048 | 0.7348 / 0.7399 / 0.7690 | 23.22 |
| 因果 | 1024 | 0.1380 / 0.1472 / 0.1724 | 29.17 |
| 因果 | 2048 | 0.4140 / 0.4207 / 0.4351 | 40.84 |

因果 `N=2048` 实际只遍历约 `51.56%` 的 Query/Key Tile 对。按 Tile 计算量估算：

\[
40.84\times51.56\%\approx21.06\ TFLOPS
\]

因此 40.84 是算法等效吞吐，约 21.06 才更接近执行 Tile 的计算吞吐，但仍不是 Profiler 给出的硬件指令吞吐。

### 12.3 显存

| N | S+P 理论容量 | Triton Naive 峰值 | Flash 峰值 |
|---:|---:|---:|---:|
| 128 | 1 MiB | 1.25 MiB | 0.25 MiB |
| 256 | 4 MiB | 4.5 MiB | 0.5 MiB |
| 512 | 16 MiB | 17 MiB | 1 MiB |
| 1024 | 64 MiB | 66 MiB | 2 MiB |
| 2048 | 256 MiB | 260 MiB | 4 MiB |

Flash 峰值与输出 `O` 的容量一致，数据直接支持“没有显式保存完整 S/P”。这不代表程序总显存是 4 MiB，因为 Q/K/V 已经在测量 baseline 中。

## 13. 从初始版本到最终版本的决策链

### 13.1 最小 correctness 版本

**实验问题：** 一个 Program 遍历全部 Key Tile，并用在线 Softmax 只输出 O，能否与 Naive 结果一致？

**结果：** 对齐、非对齐、多 dtype、因果/非因果和边界测试通过。

**决策：** 先保留清晰的单 Kernel 结构，不加入复杂 heuristic。

### 13.2 Tile 和 warp 调优

**控制变量：** 固定 `B=2,H=8,D=64,FP16`，比较不同 `BLOCK_M/BLOCK_N/num_warps`，每个候选先做 correctness。

**预测：** 更大的 Query Tile 可能提高复用，但过大的 Tile 或过多 warps 可能增加资源压力。

**结果：** `64×32, 4 warps` 在不同 N 和因果模式下总体稳定；8 warps 多数较差；`64×64` 在部分非因果点略好，但没有形成足够稳定的统一优势。

**决策：** 选择 `64×32, 4 warps` 作为统一默认，而不是为少数点增加未经充分验证的 Shape heuristic。

### 13.3 三轮随机顺序复测

**实验问题：** 单轮最佳是否只是测量顺序或 GPU 状态造成的偶然值？

**方法：** 候选配置随机顺序运行三轮，记录每轮 P20/P50/P80、最小值、最大值和相对 baseline 加速。

**结果：** `64×32,4 warps` 仍是覆盖范围更均衡的选择；小序列波动明显，`64×64` 的局部优势不足以支持复杂分支。

**决策：** 正式入口固定 `64×32,4 warps`。

### 13.4 因果跳块

**实验问题：** 完全位于未来的 Key Tile 已知全被 mask，是否值得直接不计算？

**控制变量：** 固定 `64×32,4 warps`，只改变“遍历全部 Key Tile”或“跳过完全未来 Tile”，随机顺序运行三轮。

| N | 实际 Tile 比例 | 全遍历 P50 | 跳块 P50 | 配对加速 |
|---:|---:|---:|---:|---:|
| 128 | 75.00% | 0.0163 ms | 0.0159 ms | 1.00x |
| 256 | 62.50% | 0.0272 ms | 0.0324 ms | 0.81x |
| 512 | 56.25% | 0.0928 ms | 0.0527 ms | 1.76x |
| 1024 | 53.13% | 0.2351 ms | 0.1323 ms | 1.78x |
| 2048 | 51.56% | 0.8487 ms | 0.5134 ms | 1.74x |

**解释：** 长序列减少了接近一半的 Tile 计算，因此获得稳定收益；小序列节省的工作不足以稳定抵消控制流和测量波动。

**决策：** 正式入口只在 `causal=True 且 N>=512` 时启用跳块。该阈值只对当前硬件和实验范围负责。

## 14. 失败实验和被推翻的 heuristic

1. **每个 Key 块分别做 Softmax 再直接相加。** 每块概率和都为 1，拼接后不再是全局 Softmax。必须使用在线 Softmax 合并统计量。

2. **Tile 越大越快。** 更大 Tile 提高复用，也会增加 Score 和 acc 的片上压力。`64×64` 没有在全部场景稳定胜出。

3. **warp 越多越快。** 8 warps 多数表现较差，更多线程不能自动抵消资源占用和调度成本。

4. **单轮最快就是最终最佳。** 小序列和部分配置波动明显，三轮随机顺序复测推翻了部分单点结论。

5. **因果跳块对所有 N 都有收益。** `N=128` 基本持平，`N=256` 甚至变慢，因此采用实验阈值。

6. **Flash 在所有 Shape 都比 Naive 快。** 非因果 `N=256` 的正式结果只有 Naive 的 `0.94x`。算法减少显存流量，不代表小规模固定开销必然占优。

## 15. 正确、可用和高性能的区别

### 算法正确

输出与 reference 在给定误差范围内一致；在线 Softmax、mask 和越界语义正确。

### 工程可用

接口有清楚的 dtype、device、Shape 和连续性约束；错误输入会明确报错；空序列和非对齐边界有测试；当前限制有记录。

### 性能优秀

需要在目标硬件、目标 Shape 和真实工作负载上稳定领先，并理解瓶颈来自计算、显存、寄存器还是调度。

当前版本已经证明长序列显存和延迟优势，但还不是跨硬件、跨 Shape 自动调优的生产级实现。

## 16. 证据边界

实验已经直接支持：

- Flash 不显式保存完整 S/P；
- 峰值中间容量不再随 `N^2` 增长；
- 长序列相对显式 Naive 有明显延迟优势；
- `64×32,4 warps` 是当前实验范围内较稳定的统一配置；
- 因果长序列跳过未来 Tile 有稳定收益。

仍需 Profiler 验证：

- 实际寄存器数量、occupancy 和是否 spill；
- L2/显存读取量以及带宽利用率；
- Tensor Core 利用率和真实执行 FLOPs；
- 各阶段指令和控制流开销；
- `N=512` 阈值能否迁移到其他 GPU、D、B/H 和 dtype。

因此可以说“数据符合更高复用、较少显存流量的解释”，不能只凭延迟曲线断言某个底层硬件事件已经发生。

## 17. 当前支持范围与限制

当前支持：

- CUDA Tensor；
- Q/K/V 同为 `[B,H,N,D]`；
- FP16、BF16、FP32，三者 dtype 必须一致；
- 最后一维连续；
- `0 <= N`，`1 <= D <= 128`；
- 因果和非因果 self-attention；
- 输出 Shape 和 dtype 与 Q 一致。

当前不支持：

- backward 和训练反向传播；
- Dropout、Attention bias、自定义 scale；
- Q 与 K/V 序列长度不同的 cross-attention；
- 返回完整 S/P 或 Softmax 统计量；
- `D>128`；
- FlashAttention 2/3 的复杂调度、持久化或多阶段优化；
- 跨设备自动选择 Tile 和跳块阈值。

## 18. 面试表达

可以用下面这段话概括本阶段：

> Naive Attention 会将形状为 `[B,H,N,N]` 的 Score 和 Probability 写入全局显存，因此中间张量容量和显存流量都随 `N²` 增长。
>
> 我实现的 Triton FlashAttention Forward 让一个 Program 固定处理一个 Query Tile，并在 Program 内依次遍历 K/V Tile。它使用 FP32 的运行最大值 `m`、归一化分母 `l` 和输出累加器 `acc` 完成在线 Softmax。
>
> 当新的 Key Tile 带来更大的最大值时，旧的 `l` 和 `acc` 都要乘：
>
> \[ \alpha=\exp(m_{old}-m_{new}) \]
>
> 这样可以把新旧结果转换到相同的指数基准。全部 Key Tile 处理完成后，通过 `O=acc/l` 得到最终输出。
>
> 这种设计将 `QK^T`、缩放、因果遮罩、Softmax 和 `P@V` 融合在一个 Kernel 中。Score 和 Probability 只以当前 Tile 的形式短暂存在，不会把完整 S/P 写回全局显存。因此，它减少的是 `N²` 中间张量容量和显存流量，计算复杂度仍然是 `O(N²D)`。
>
> 在性能调优方面：我先以 `32×32、4 warps` 为 baseline，控制变量测试了不同的 Query Tile、Key Tile 和 warp 数。所有候选配置先通过 correctness，然后进行 Benchmark。第一轮实验显示 `64×32、4 warps` 整体表现较好，我又通过三轮随机执行顺序复测，降低测量顺序和 GPU 状态的影响。虽然 `64×64` 在少数非因果 Shape 上略快，但没有形成稳定的跨 Shape 优势；8 warps 在多数场景表现较差。因此，最终选择 `64×32、4 warps` 作为统一配置，而没有加入容易过拟合的复杂 heuristic。
>
> 对于因果 Attention，我还单独比较了“遍历全部 Key Tile”和“跳过完全位于当前 Query Tile 未来的 Key Tile”。实验发现，小序列收益不稳定，但当N > 512时，才开始有收益，具体体现在加速比上，因此我选择只在 `causal=True 且 N>=512` 时启用未来 Key Tile 跳过
>
> 最终结果表明，FlashAttention 在长序列上获得了明显的延迟和显存优势。不过，目前的数据只能直接证明性能现象；寄存器占用、occupancy、显存带宽利用率和 Tensor Core 利用率等底层原因，仍需要 Nsight Compute 和生成指令进一步验证。

## 19. 最终总结

FlashAttention 最重要的思想不是一个难记的公式，而是三件事：

1. **分块计算：** 当前只保留一个小 Score Tile，不保存完整 S/P。
2. **在线合并：** 用 `m、l、acc` 把多个 Key Tile 合并成与整行 Softmax 相同的结果。
3. **数据复用：** 一个 Program 复用 Q Tile，连续完成 Score、Softmax 和输出累加，只在最后写 O。

