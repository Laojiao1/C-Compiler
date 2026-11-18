# Triton 学习阶段复盘

> 当前学习范围：GPU 基础性能模型、Triton 编程模型、Vector Add、Fused Softmax、Online Softmax，以及 Online Softmax 到 FlashAttention 的思想衔接。

## 1. 当前所处阶段

目前已经完成并实际验证的内容包括：

- `01_vector_add`：正确性测试、输入检查、边界处理、Benchmark、有效带宽分析和 `BLOCK_SIZE` 对比。
- `02_softmax/softmax`：普通 Fused Softmax 的实现、正确性测试和性能分析。
- `02_softmax/softmax_online`：Online Softmax 的数学原理、Triton 实现、正确性测试和 Benchmark。
- `02_softmax/softmax_FlashAtten`：用 Online Softmax 状态更新推导 `softmax(X) @ V`，理解它与 FlashAttention 的联系。

下一阶段计划学习 Matrix Multiplication，但在进入新算子之前，应先确认本笔记中的核心问题都能独立回答。

---

## 2. GPU 执行模型

### 2.1 CPU 与 GPU 的基本差异

CPU 擅长复杂控制流、低延迟和少量强线程；GPU 擅长让大量执行单元同时处理结构相似的数据。

写 GPU 算子时，核心目标通常不是让单个计算步骤最快，而是：

- 暴露足够多的并行任务；
- 让显存访问连续并尽量合并；
- 减少不必要的全局显存读写；
- 合理使用寄存器和片上存储；
- 根据算子的瓶颈选择优化方向。

### 2.2 SM 流式多处理器

SM（Streaming Multiprocessor）是 NVIDIA GPU 执行计算任务的主要硬件单元。

一个 GPU 包含多个 SM。程序启动后，大量工作块会被调度到这些 SM 上执行。一个 SM 可以同时驻留多个任务，但能够驻留多少任务受以下资源限制：

- 寄存器数量；
- Shared Memory 容量；
- 最大线程数和 Warp 数；
- 硬件允许的最大 Block 数量。

当前设备为 RTX 5060 Laptop GPU，共有 26 个 SM。因此，如果一个算子只启动很少的 Triton program，例如只有 16 行数据且每行对应一个 program，就无法充分利用全部 SM。

### 2.3 Warp 线程束

Warp 是 NVIDIA GPU 的基本线程调度单位，通常由 32 个线程组成。一个 Warp 中的线程以 SIMT 方式执行相同指令。

当同一个 Warp 内的线程走不同分支时，可能产生分支分歧：不同路径通常需要分开执行，从而降低有效并行度。

Triton 不要求我们直接用 CUDA 线程编号组织算法，但生成的程序最终仍会映射到 Warp 和 GPU 硬件上，所以 `num_warps`、tile 大小、分支和资源使用仍然会影响性能。

### 2.4 Triton Program Instance

Triton 的核心抽象不是“一个线程处理一个元素”，而是：

> 一个 program instance 负责处理一块数据。

例如 Vector Add 中：

- 一个 program 处理 `BLOCK_SIZE` 个元素；
- `tl.program_id(0)` 表示当前 program 在第 0 维网格中的编号；
- program 根据编号计算自己负责的起始位置。

Softmax 中通常采用：

- 一行对应一个 program；
- program 内部一次处理或分块遍历这一行；
- 不同行之间并行执行。

这就是 Triton 的 Blocked Program 思维。

### 2.5 Grid

Grid 决定启动多少个 program instance。

Vector Add 的典型形式为：

```python
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
```

Softmax 的典型形式为：

```python
grid = (n_rows,)
```

Grid 过小会导致 GPU 并行度不足；Grid 足够大时，各个 SM 才能持续获得工作。

---

## 3. GPU 存储层级

### 3.1 Global Memory

Global Memory 即 GPU 显存：

- 容量最大；
- 延迟较高；
- 所有 SM 都可访问；
- 连续大规模访问时可以获得较高带宽。

Vector Add 和 Softmax 都需要从 Global Memory 读取输入并写回输出。对于 memory-bound 算子，减少显存流量往往比减少少量计算更重要。

### 3.2 L2 Cache

L2 Cache 位于显存和 SM 之间，由整个 GPU 共享。重复访问的数据可能命中 L2，从而避免再次访问更慢的显存。

Benchmark 中小规模输入可能主要落在缓存里，因此测出的“有效带宽”有时不能直接等同于真实 DRAM 带宽。分析带宽平台时，应优先观察足够大的输入规模。

### 3.3 Shared Memory / SRAM

Shared Memory 是 SM 上的片上存储：

- 延迟和带宽优于 Global Memory；
- 容量有限；
- 由同一线程块内的线程共享；
- 使用过多会限制同时驻留在 SM 上的任务数量。

在 Triton 中，我们通常描述 tile 和计算过程，由编译器决定数据放在寄存器、Shared Memory 或其他位置。不能简单地认为所有临时变量都必然在某一种存储中。

### 3.4 Register

寄存器是执行单元附近最快的存储资源，通常用于保存：

- 局部变量；
- 中间计算结果；
- reduction 状态；
- MatMul 中的 accumulator。

寄存器容量有限。单个 program 使用的寄存器太多时，会出现 Register Pressure。

严重时可能发生寄存器溢出（spill），把本应放在寄存器中的数据放到更慢的存储中，导致性能下降。

### 3.5 存储层级的核心认识

不要只记“越靠近计算单元越快”，还要同时考虑：

- 容量有限；
- 资源由同时运行的任务共享；
- tile 越大，中间状态通常越多；
- 片上资源使用过多可能降低并发度；
- 全局显存访问次数和访问模式会直接决定 memory-bound 算子的性能。

---

## 4. Occupancy 占用与资源压力

### 4.1 Occupancy

Occupancy 可以粗略理解为：一个 SM 上实际驻留的活跃 Warp 数量，相对于硬件允许最大活跃 Warp 数量的比例。

较高 Occupancy 有助于隐藏访存和指令延迟：当一个 Warp 等待数据时，SM 可以切换去执行另一个 Warp。

但 Occupancy 不是越高越好，也不是性能的唯一目标。只要已经有足够并行度隐藏延迟，继续提高 Occupancy 未必带来收益。

### 4.2 影响 Occupancy 的因素

- 每个 program 使用的寄存器数量；
- Shared Memory 使用量；
- `num_warps`；
- tile / `BLOCK_SIZE` 大小；
- 硬件最大驻留任务限制。

### 4.3 BLOCK_SIZE 的权衡

增大 `BLOCK_SIZE` 的潜在好处：

- program 数量减少；
- 循环次数可能减少；
- 固定调度和索引开销被更多数据摊薄。

潜在坏处：

- 单个 program 处理的数据更多；
- 寄存器和片上资源压力增大；
- 可能降低 Occupancy；
- 极端情况下发生 spill；
- 大量无效位置可能被 mask，浪费计算资源。

因此不能认为 `BLOCK_SIZE` 越大越快，必须通过实验判断。

---

## 5. 访存合并、Stride 与连续布局

### 5.1 Coalesced Memory Access

访存合并是 GPU 性能中的生死线概念。

当相邻执行 lane 访问相邻、对齐的内存地址时，硬件可以把多个访问合并为较少的内存事务，从而充分利用显存带宽。

Vector Add 中的访问方式：

```text
x[offsets]
y[offsets]
output[offsets]
```

其中 `offsets` 连续递增，因此天然适合合并访问。

### 5.2 Stride

Stride 表示沿某个维度移动一个元素时，在底层存储中需要跨过多少个元素。

二维矩阵地址通常可表示为：

```text
address(row, col) = base + row * stride_row + col * stride_col
```

如果最后一维连续，则 `stride_col == 1`。Row-wise Softmax 通常要求最后一维连续，这样同一行的数据才能被连续加载。

### 5.3 为什么非连续输入需要明确处理

如果 kernel 按连续内存编写，却接收到最后一维非连续的张量，就可能读取错误位置。

工程上有两种选择：

1. 支持任意 stride，并把 stride 作为 kernel 参数；
2. 明确限制输入布局，在 Python 包装层拒绝不支持的输入（强制检查连续）

当前 Softmax 选择第二种方式，这属于清晰的输入契约，而不是功能缺陷。

### 5.4 Alignment 对齐

地址对齐会影响内存事务是否高效。连续访问不一定自动意味着完美对齐，但连续、规则、适当对齐的访问通常最容易获得高带宽。

---

## 6. Roofline 性能模型

### 6.1 两类主要瓶颈

算子性能通常受到两种上限之一限制：

- Memory-bound：受显存带宽限制；
- Compute-bound：受计算吞吐量限制。

Vector Add 和 Softmax 通常偏 memory-bound；大型 MatMul 通常更可能偏 compute-bound。

### 6.2 Arithmetic Intensity 算数强度

算术强度定义为：

```text
Arithmetic Intensity = 计算量 FLOPs / 数据搬运量 Bytes
```

Vector Add 每个元素只做一次加法，却需要读取两个输入并写入一个输出，因此算术强度很低。

MatMul 会重复使用已经加载的 tile，单次数据搬运可以支持大量乘加，因此算术强度较高。

### 6.3 Effective Bandwidth 有效带宽

对 FP32 Vector Add，每个元素的算法最小流量为：

- 读取 `x`：4 Bytes；
- 读取 `y`：4 Bytes；
- 写入 `output`：4 Bytes。

共 12 Bytes，因此：

```text
Effective GB/s = 12 * N / time / 1e9
```

这里的 Effective Bandwidth 是基于算法所需流量计算出的有效指标，不一定等于硬件实际 DRAM 事务量。

### 6.4 TFLOPS

对于计算密集型算子，应使用计算吞吐量：

```text
TFLOPS = FLOPs / time / 1e12
```

MatMul `C = A @ B` 的近似计算量为：

```text
2 * M * N * K FLOPs
```

这是下一阶段分析 MatMul 时的核心性能指标。

### 6.5 优化方向必须服从瓶颈

如果算子是 memory-bound：

- 减少 Global Memory 读写；
- 改善访存合并；
- 进行算子融合；
- 避免不必要的中间张量。

如果算子是 compute-bound：

- 提高 Tensor Core 利用率；
- 改善 tile 设计和数据复用；
- 减少无效计算；
- 让计算流水线更加充分。

---

## 7. Triton 编程模型与常用 API

### 7.1 `@triton.jit`

用于声明 Triton JIT kernel。第一次遇到新的 shape、dtype 或编译期参数组合时，Triton 可能进行编译和缓存。

### 7.2 `tl.program_id(axis)`

取得当前 program 在指定 grid 维度上的编号。

Vector Add 使用一维编号确定数据块；二维 MatMul 将使用二维或经过重排的一维编号确定输出 tile。

### 7.3 `tl.arange`

生成一个连续索引向量，例如：

```python
offsets = block_start + tl.arange(0, BLOCK_SIZE)
```

它表达的是一个 program 内部同时操作的一组位置，而不是 Python 中逐元素执行的普通循环。

### 7.4 `tl.load` 与 `tl.store`

用于从内存加载和写回数据：

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
tl.store(output_ptr + offsets, result, mask=mask)
```

### 7.5 Mask

当数据长度不是 `BLOCK_SIZE` 的整数倍时，最后一个 program 会产生越界索引。Mask 用于阻止这些位置真正访问内存。

Vector Add 的典型 mask：

```python
mask = offsets < n_elements
```

Reduction 中，被 mask 的位置还必须选择正确的 `other` 值：

- 求最大值时通常填 `-inf`；
- 求和时通常填 `0`。

Mask 不只是“防止程序崩溃”，还必须保证无效元素不会改变数学结果。

### 7.6 `tl.max` 与 `tl.sum`

它们用于在一个 tile 的指定维度上进行归约。

Softmax 需要：

- `tl.max` 计算行最大值；
- `tl.sum` 计算指数和。

Reduction 通常比逐元素运算更复杂，因为不同 lane 的局部结果需要被合并。

### 7.7 编译期常量

使用 `tl.constexpr` 的参数在编译期间已知，例如 `BLOCK_SIZE`。不同的编译期参数通常会生成不同版本的 kernel。

这使编译器能够展开循环和进行针对性优化，但配置过多也会增加编译和调优成本。

---

## 8. Vector Add 复盘

### 8.1 数学定义

```text
output[i] = x[i] + y[i]
```

各元素之间完全独立，因此天然适合数据并行。

### 8.2 数据切分

每个 program 处理 `BLOCK_SIZE` 个连续元素：

```text
block_start = program_id * BLOCK_SIZE
offsets = block_start + [0, 1, ..., BLOCK_SIZE - 1]
```

总 program 数为：

```text
ceil(N / BLOCK_SIZE)
```

### 8.3 边界处理

当 `N` 不能整除 `BLOCK_SIZE` 时，最后一个 program 只处理其中一部分位置：

```text
mask = offsets < N
```

### 8.4 性能本质

Vector Add 的计算量很小，主要时间用于：

- 从显存读取 `x`；
- 从显存读取 `y`；
- 将结果写回显存。

因此它是典型 memory-bound 算子，主要观察 Effective GB/s，而不是 FLOPS。

### 8.5 BLOCK_SIZE 实验结论

已经比较过 `256 / 512 / 1024 / 2048`：

- 小规模输入中，固定开销和测量噪声占比较高，各配置会有波动；
- 大规模输入中，各配置最终收敛到约 344～346 GB/s；
- 配置差距不足以证明需要引入复杂 autotune；
- 当前默认选择 `BLOCK_SIZE=1024` 是合理的工程折中。

这个实验说明：

> 对已经触及带宽平台的 memory-bound 算子，仅调整 BLOCK_SIZE 通常无法突破硬件带宽上限。

### 8.6 Vector Add 应掌握的问题

- 为什么 grid 要用向上取整？
- 为什么最后一个 block 需要 mask？
- 为什么 FP32 Vector Add 按每元素 12 Bytes 计算有效带宽？
- 为什么大规模输入时不同 BLOCK_SIZE 的性能趋同？
- 为什么 Triton 只与 PyTorch 接近，而不是大幅领先？

---

## 9. Softmax 数学与数值稳定性

### 9.1 Softmax 定义

对一行输入 `x`：

```text
softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
```

### 9.2 数值溢出问题

直接计算 `exp(x_i)` 时，如果 `x_i` 很大，指数可能溢出。

利用 Softmax 的平移不变性：

```text
softmax(x_i)
= exp(x_i - m) / sum_j(exp(x_j - m))
```

其中：

```text
m = max_j(x_j)
```

因为所有 `x_i - m <= 0`，最大指数为 1，从而显著降低溢出风险。

### 9.3 为什么减去任意常数都不改变结果

```text
exp(x_i - c) / sum_j(exp(x_j - c))
= exp(x_i) * exp(-c) / [exp(-c) * sum_j(exp(x_j))]
= exp(x_i) / sum_j(exp(x_j))
```

选择最大值作为 `c`，是因为它能让最大指数恰好等于 1。

---

## 10. Fused Softmax

### 10.1 为什么称为 Fused

朴素实现可能分成多个 kernel：

1. 求最大值；
2. 计算指数；
3. 求指数和；
4. 除法归一化。

每一步之间都可能产生 Global Memory 中间张量。

Fused Softmax 让一个 program 在片上完成一整行的最大值、指数、求和和归一化，只在开始读取输入并在最后写回输出。

### 10.2 典型执行过程

对每一行：

1. 读取整行输入；
2. 求行最大值；
3. 计算 `exp(x - max)`；
4. 求指数和；
5. 计算概率；
6. 写回输出。

### 10.3 为什么通常补齐到 2 的幂

Triton reduction 常使用大于等于列数的下一个 2 的幂作为 `BLOCK_SIZE`：

```text
BLOCK_SIZE = next_power_of_2(N)
```

真实元素用 mask 加载，其余位置用不会干扰归约的值填充。

这有利于生成规则的 reduction（归约），但当 `N` 刚超过一个 2 的幂时，可能出现大量 padding，例如 `N=2049` 时需要 `BLOCK_SIZE=4096`。

### 10.4 Mask 填充值

求最大值时，无效位置应填：

```text
-infinity
```

否则无效的 0 可能错误地成为最大值。

### 10.5 显存流量

理想 Fused Softmax 对 FP32 每个元素：

- 读取输入：4 Bytes；
- 写回输出：4 Bytes。

算法最小流量约为：

```text
8 * M * N Bytes
```

相比拆成多个 PyTorch eager 操作的朴素实现，它避免了多个中间张量的反复读写。

### 10.6 性能特点

- 相比朴素 eager 实现通常有明显优势；
- 与 PyTorch 高度优化实现相比，优势依赖 shape；
- 某些列数附近可能因为 `BLOCK_SIZE` 跨越 2 的幂而出现性能变化；
- 列数过大会带来严重的片上资源压力，单块整行方案不再适用。

---

## 11. Online Softmax

### 11.1 为什么需要 Online Softmax

标准 safe softmax 通常需要：

1. 第一遍求全局最大值；
2. 第二遍求 `sum(exp(x - max))`；
3. 第三遍读取输入并生成概率。

Online Softmax 的目标是：

> 在一遍分块扫描中同时维护当前最大值和经过正确缩放的指数和。

这样第一阶段只需一遍扫描即可获得全局最大值与分母。

### 11.2 状态不变量

处理完前缀数据后，维护：

```text
m_i = 当前已处理元素的最大值
l_i = sum(exp(x_j - m_i))，其中 j 属于已处理前缀
```

关键点是：`l_i` 不是普通的 `sum(exp(x))`，它始终以当前最大值 `m_i` 为基准。

### 11.3 合并新 Chunk

设新 chunk 的最大值为 `m_chunk`，新的共同最大值为：

```text
m_new = max(m_i, m_chunk)
```

旧的指数和原本以 `m_i` 为基准，现在必须换成以 `m_new` 为基准：

```text
alpha = exp(m_i - m_new)
```

更新公式：

```text
l_new = l_i * alpha + sum(exp(x_chunk - m_new))
```

然后令：

```text
m_i = m_new
l_i = l_new
```

### 11.4 为什么旧状态必须乘 alpha

旧状态为：

```text
l_i = sum(exp(x_j - m_i))
```

要改写到新的最大值 `m_new` 下：

```text
exp(x_j - m_new)
= exp(x_j - m_i) * exp(m_i - m_new)
```

因此全部旧贡献都必须乘：

```text
alpha = exp(m_i - m_new)
```

如果新 chunk 没有更大的值，则 `m_new = m_i`，此时 `alpha = 1`，旧状态不变。

### 11.5 为什么仍需要第二遍读取

第一遍结束后得到最终的：

```text
m = 全局最大值
l = sum(exp(x - m))
```

但要输出每个元素的概率，仍需知道每个 `x_i`：

```text
y_i = exp(x_i - m) / l
```

如果没有把整行输入保存在片上，就需要第二遍读取输入。

因此当前 Online Softmax 是两遍：

1. 第一遍分块维护 `m/l`；
2. 第二遍重新读取输入并写出概率。

### 11.6 空间复杂度

持久状态只有 `m_i` 和 `l_i`，因此算法状态空间为 `O(1)`。

但一次仍会加载一个大小为 `BLOCK_SIZE` 的 chunk，所以当前临时 tile 资源为 `O(BLOCK_SIZE)`。

“算法状态是 O(1)”不代表执行过程中完全不需要 tile。

### 11.7 显存流量比较

以 FP32 为例：

| 实现 | 主要流量 | 近似算法流量 |
|---|---|---:|
| Fused Softmax | 读输入一次、写输出一次 | `8MN` Bytes |
| Online Softmax | 第一遍读输入、第二遍读输入、写输出 | `12MN` Bytes |
| 传统三遍 Safe Softmax | 三次读输入、写输出 | `16MN` Bytes |

Online Softmax 减少了扫描次数，但对能够整行放入片上的 shape，它不一定比 Fused Softmax 更快。

### 11.8 当前实现的并行结构

当前 Online Softmax：

- `grid = (M,)`；
- 每行由一个 program 负责；
- 行内通过循环按 chunk 串行扫描；
- 循环次数约为 `ceil(N / BLOCK_SIZE)`。

当 `M` 很小而 `N` 很大时：

- program 总数太少；
- 无法充分占满所有 SM；
- 单个 program 内部循环很长；
- 行内存在串行依赖。

因此即使总元素数量相同，少量超长行也可能明显慢于大量较短行。

### 11.9 已观察到的性能现象

固定总元素数量时：

- `4096 × 4096`：行数多、program 多、每行 chunk 少；
- `16 × 1048576`：行数少、program 少、每行需要约 1024 个 chunk。

后者明显更慢，说明仅用总元素数量不能解释性能，还必须考虑：

- 并行 program 数量；
- 单个 program 内循环长度；
- 状态依赖链；
- Occupancy 与资源压力。

---

## 12. 从 Online Softmax 到 FlashAttention

### 12.1 普通 Attention

Attention 的核心形式为：

```text
S = QK^T
P = softmax(S)
O = PV

最终结果 = softmax(QK^T) * V
```

朴素实现会把巨大的中间矩阵 `S` 和 `P` 写入 Global Memory。

FlashAttention 的核心方向是：

> 分块计算分数，在片上维护 Online Softmax 状态，并同步累积输出，避免完整物化注意力矩阵。

### 12.2 当前桥接实验维护的状态

当前教学实验处理的是：

```text
O = softmax(X) @ V
```

它维护三类状态：

```text
m   = 当前最大值
l   = 当前经过缩放的指数和
acc = 当前经过相同缩放的加权输出累积
```

### 12.3 acc 更新公式

当最大值更新为 `m_new` 时：

```text
alpha = exp(m_old - m_new)
p_chunk = exp(x_chunk - m_new)
```

不仅旧的 `l` 要乘 `alpha`，旧的 `acc` 也必须乘相同的 `alpha`：

```text
l_new   = l_old * alpha + sum(p_chunk)
acc_new = acc_old * alpha + sum(p_chunk[:, None] * V_chunk, axis=0)
```

最终输出：

```text
O = acc / l
```

### 12.4 为什么这能避免存储完整概率

每个 chunk 的概率贡献被立刻乘到对应的 `V` 上并加入 `acc`，不需要把完整的 Softmax 概率向量写入 Global Memory。

### 12.5 当前实验不等于完整 FlashAttention

当前实验只是思想桥梁，因为它仍然接收已经存在的 `X`。完整 FlashAttention 还需要：

- 分块加载 `Q` 和 `K`；
- 在片上计算 `QK^T` tile；
- 应用缩放因子；
- 支持 causal mask；
- 处理 batch 和 head 维度；
- 不物化完整 `QK^T`；
- 设计 backward。

因此目前掌握的是 FlashAttention 最核心的 Online Softmax 状态合并思想，而不是完整工程实现。

---

## 13. 输入契约与工程边界

算子不仅要在正常输入上算对，还要明确自己支持什么。

当前已经涉及的输入检查包括：

- 输入张量必须在 CUDA 上；
- Vector Add 两个输入 shape 必须一致；
- 两个输入 dtype 必须一致；
- Softmax 输入维度必须符合接口约定；
- Softmax 最后一维必须连续；
- Online Softmax 的 `BLOCK_SIZE` 必须符合实现限制；
- Flash bridge 中输入 shape 必须兼容；
- 空张量需要安全返回。

输入检查的意义：

1. 防止错误输入造成越界或错误结果；
2. 让接口行为可预测；
3. 把 kernel 的实现假设明确写成契约；
4. 让测试覆盖正常路径和失败路径。

---

## 14. 正确性测试方法

### 14.1 为什么不能只测试一个 Shape

只测试整齐的大 shape 很容易掩盖边界错误。测试应覆盖：

- 空输入；
- 极小输入；
- 正好整除 `BLOCK_SIZE`；
- 不能整除 `BLOCK_SIZE`；
- 2 的幂边界附近，例如 `63 / 64 / 65`；
- 不同 dtype；
- 非连续布局；
- 不兼容 shape；
- 不合法参数。

### 14.2 参考实现

常用 PyTorch 运算作为正确性参考：

- Vector Add：`torch.add` 或 `x + y`；
- Softmax：`torch.softmax`；
- `softmax(X) @ V`：PyTorch 组合表达式。

### 14.3 浮点误差

浮点数运算不满足严格结合律。不同 reduction 顺序可能产生微小差异，因此通常使用：

```python
torch.testing.assert_close(..., atol=..., rtol=...)
```

容差应根据 dtype、算法和累加精度设定，不能机械地要求逐 bit 相等。

### 14.4 负向测试

除了验证合法输入，还应确认错误输入会被明确拒绝。例如：

- dtype 不一致；
- shape 不兼容；
- 不支持的 stride；
- 非法 `BLOCK_SIZE`。

一个工程可用的接口必须既会“正确计算”，也会“正确失败”。

---

## 15. Benchmark 方法

### 15.1 Warmup

第一次调用可能包括：

- JIT 编译；
- CUDA context 初始化；
- 内存分配和缓存建立；
- GPU 频率爬升。

因此正式计时前需要 warmup，避免把一次性成本混入稳态性能。

### 15.2 重复测量与分位数

单次计时容易受到系统噪声影响。应重复测量并报告：

- P20：较快但不是极端最小值；
- P50：中位数，代表典型表现；
- P80：偏慢表现，可观察稳定性。

曲线阴影范围较宽时，通常说明测量抖动较大，不能仅凭单个异常点下结论。

### 15.3 Kernel Time 与算子级延迟

如果 Python 包装函数内部创建输出张量，那么计时可能包含输出分配成本。

因此必须明确指标是：

- 纯 kernel time；还是
- 包含输出张量分配的算子级延迟。

两者都可以有意义，但不能混为一谈。

### 15.4 控制变量

优化实验应一次尽量只改变一个变量，例如：

- 只改变 `BLOCK_SIZE`；
- 只改变 `num_warps`；
- 固定总元素数量，只改变矩阵形状；
- 固定行数，只改变列数。

否则即使性能变化，也很难知道是哪个因素造成的。

### 15.5 不要过度解释噪声

遇到单个 shape 的异常点时，应先：

1. 重复运行；
2. 观察多个分位数；
3. 检查是否跨越编译配置边界；
4. 检查 GPU 温度、功耗和后台负载；
5. 再提出原因假设。

性能分析需要证据，不能仅凭一条曲线讲故事。

---

## 16. 当前容易混淆的概念

### 16.1 Program 不等于 CUDA Thread

Triton program 负责一个数据块，内部的向量操作最终由编译器映射到多个线程和 Warp。

### 16.2 BLOCK_SIZE 不等于线程数

`BLOCK_SIZE` 描述一个 program 处理的逻辑元素数量，不应直接理解成 CUDA Block 中的线程数量。

### 16.3 O(1) 状态不等于 O(1) 临时资源

Online Softmax 只持久维护 `m/l` 两个状态，但当前 chunk 仍需占用与 `BLOCK_SIZE` 有关的片上资源。

### 16.4 Fused 不一定永远更快

融合减少了显存中间读写，但可能增加单个 kernel 的资源压力。是否更快仍需结合 shape、资源使用和硬件验证。

### 16.5 Occupancy 高不等于性能一定高

Occupancy 是隐藏延迟的手段，不是最终目标。访存模式、计算吞吐和数据复用同样关键。

### 16.6 BLOCK_SIZE 大不等于性能高

更大的块可能减少循环和调度开销，也可能带来更高寄存器压力、更低并发度和更多 padding。

### 16.7 Effective GB/s 不等于硬件标称显存带宽

它是按算法理论流量计算出的性能指标，实际内存事务、缓存命中和写策略可能不同。

### 16.8 算法正确不等于工程可用

工程可用还需要：

- 明确输入契约；
- 完整测试；
- 可重复 Benchmark；
- 合理错误信息；
- 适用范围说明；
- 多 shape 和 dtype 支持。

---

## 17. 当前知识缺口

以下内容将在 MatMul 及后续算子中继续学习：

### 17.1 MatMul 相关

- 二维 tiling；
- `BLOCK_M / BLOCK_N / BLOCK_K`；
- 指针广播；
- K 维循环；
- accumulator；
- `tl.dot`；
- Tensor Core；
- FP16/BF16 输入与 FP32 累加；
- 不同 shape 下的 tile 选择；
- TFLOPS 与计算效率。

### 17.2 更深入的性能工具

- 查看编译后的 PTX / SASS；
- 分析寄存器数量；
- 检查 spill；
- 使用 profiler 分析内存吞吐、Occupancy 和 Tensor Core 利用率；
- 区分理论推断与 profiler 证据。

### 17.3 后续算子

- LayerNorm：更多 reduction 与融合；
- Naive Attention：组合 MatMul、Softmax 和数据布局；
- FlashAttention：分块 Attention、Online Softmax 与 IO-aware 算法；
- Backward：梯度公式、保存状态与重计算权衡。

---

## 18. 推荐的复盘方式

不要只阅读代码，应按以下顺序主动复盘每个算子：

1. 先写出数学定义；
2. 画出每个 program 负责的数据范围；
3. 手算一个很小的输入；
4. 标出每次 Global Memory 读取和写回；
5. 说明每个 mask 的作用和填充值；
6. 估算算法流量或 FLOPs；
7. 判断 memory-bound 还是 compute-bound；
8. 预测 shape 或参数变化会造成什么现象；
9. 用测试和 Benchmark 验证预测；
10. 记录预测与结果不一致的地方。

推荐学习循环：

```text
最少必要理论
    ↓
手写或补全 Kernel
    ↓
正确性测试
    ↓
Benchmark
    ↓
提出性能假设
    ↓
一次只改一个变量进行验证
    ↓
整理笔记和面试表达
```

---

## 19. 查漏补缺自测清单

### 19.1 GPU 基础

- [x] 能解释 SM 和 Warp 分别是什么。
- [x] 能解释 Triton program 与 CUDA thread 的区别。
- [x] 能解释 Grid 决定了什么。
- [x] 能按速度、容量和作用解释 Global Memory、L2、Shared Memory、Register。
- [x] 能解释 Register Pressure 和 spill 的后果。
- [x] 能解释 Occupancy 的作用，以及为什么它不是越高越好。
- [x] 能解释访存合并为什么重要。
- [x] 能根据 stride 判断一个维度是否连续。

### 19.2 性能模型

- [x] 能解释 arithmetic intensity。
- [x] 能判断 Vector Add 为什么是 memory-bound。
- [x] 能解释 MatMul 为什么更可能是 compute-bound。
- [x] 能计算 FP32 Vector Add 的有效带宽。
- [x] 能说明 Effective GB/s 与真实 DRAM 带宽的区别。
- [x] 能解释为什么优化方向必须先服从瓶颈类型。

### 19.3 Triton 基础

- [x] 能独立写出 `tl.program_id`、`tl.arange` 和 offsets。
- [x] 能解释 `triton.cdiv` 为什么用于计算 Grid。
- [x] 能正确使用 `tl.load`、`tl.store` 和 mask。
- [x] 能为 max reduction 与 sum reduction选择正确的 mask 填充值。
- [x] 能解释 `tl.constexpr` 的作用。
- [x] 能说明 `BLOCK_SIZE` 为什么不是越大越好。

### 19.4 Vector Add

- [x] 能不看代码写出 Vector Add kernel 的结构。
- [x] 能处理 `N=0` 和不能整除 `BLOCK_SIZE` 的情况。
- [x] 能说明输入包装层应检查哪些条件。
- [x] 能解释不同 BLOCK_SIZE 在大输入下为什么趋于同一带宽平台。
- [x] 能解释为什么 Triton 与 PyTorch 性能接近是合理结果。

### 19.5 Fused Softmax

- [x] 能推导减去最大值不改变 Softmax 结果。
- [x] 能解释为什么减最大值提高数值稳定性。
- [x] 能说清一行对应一个 program 的执行流程。
- [x] 能解释为什么使用 `next_power_of_2`。
- [x] 能解释 padding 过多可能造成的性能问题。
- [x] 能比较 Fused Softmax 与朴素 eager 实现的显存流量。

### 19.6 Online Softmax

- [x] 能准确说出 `m_i` 与 `l_i` 的含义。
- [x] 能独立写出 `m_new`、`alpha` 和 `l_new` 的更新公式。
- [x] 能推导旧 `l_i` 为什么必须乘 `alpha`。
- [x] 能解释为什么 Online Softmax 仍需要第二遍读取才能输出概率。
- [x] 能区分持久状态 `O(1)` 与临时 tile `O(BLOCK_SIZE)`。
- [x] 能解释少量超长行为什么可能比大量短行更慢。
- [x] 能比较 Fused、Online 和传统三遍 Softmax 的算法流量。

### 19.7 FlashAttention 衔接

- [x] 能写出 `m/l/acc` 三个状态。
- [x] 能解释最大值变化时为什么 `acc` 也要乘 `alpha`。
- [x] 能解释为什么边计算概率边乘 `V` 可以避免存储完整概率矩阵。
- [x] 能说明当前 `softmax(X) @ V` 实验与完整 FlashAttention 的差距。

### 19.8 测试与 Benchmark

- [x] 能设计普通、边界和错误输入测试。
- [x] 能解释为什么使用浮点容差而不是严格相等。
- [x] 能解释 warmup、重复测量和 P20/P50/P80。
- [x] 能区分纯 kernel time 与包含输出分配的算子级延迟。
- [x] 能坚持一次只改变一个性能变量。
- [x] 能避免根据单个异常点仓促得出结论。

---

## 20. 阶段性结论

目前最重要的进步不是记住了几个 Triton API，而是已经建立了一套分析算子的基本方法：

1. 从数学定义确认结果应该是什么；
2. 从 program 和 tile 角度划分工作；
3. 从存储层级分析数据在哪里移动；
4. 从显存流量和 FLOPs 判断性能瓶颈；
5. 用边界测试建立正确性底线；
6. 用可重复 Benchmark 验证性能假设；
7. 根据证据进行单变量优化。

如果能够独立回答自测清单中的大部分问题，就已经具备进入 MatMul 学习阶段的基础。下一阶段不需要先把所有 MatMul 理论背完，而应继续沿用“小步闭环”的方法，在编写二维 tiled MatMul 的过程中逐步理解数据复用、`tl.dot`、Tensor Core 和 compute-bound 性能分析。
