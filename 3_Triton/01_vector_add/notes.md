# 01_vector_add 复盘

## 1. 算子与支持范围

- 公式：`output = x + y`
- 当前支持：CUDA、连续张量、相同 shape/device/dtype，dtype 为 FP16、BF16 或 FP32。
- Python 封装把任意维连续张量展平成一维地址空间；Kernel 不关心原始逻辑维度。
- 空张量直接返回，不启动大小为 0 的 grid。

## 2. Triton 映射

每个 program instance 负责 `BLOCK_SIZE` 个连续元素：

```text
pid
 -> block_start = pid * BLOCK_SIZE
 -> offsets = block_start + [0, ..., BLOCK_SIZE - 1]
 -> mask = offsets < n_elements
 -> load x/y, add, store output
```

连续 `offsets` 是合并访存的基础。最后一个 tile 使用同一份 mask 保护 load 和 store。

`tl.load` 产生的是 Triton blocked tensor，编译器通常将这些短生命周期值映射到寄存器。这里没有显式申请 CUDA Shared Memory，也没有跨 program 的数据共享。

## 3. 性能模型

Vector Add 的算术强度很低：每个元素只有一次加法，但至少需要两个输入和一个输出，因此通常是 memory-bound。

对于 FP32，按算子语义计算的最小数据量为：

```text
读取 x：4 Bytes
读取 y：4 Bytes
写入 output：4 Bytes
总计：12 Bytes / element
```

逻辑有效带宽：

```text
effective_GB/s = 12 * N / latency_ms * 1e-6
```

这里的 12 Bytes 是算法字节量，不是硬件计数器测得的 DRAM 流量。真实流量还可能受到 L2 cache、写策略、内存事务粒度和 benchmark 重复访问的影响。

## 4. Correctness 闭环

当前测试覆盖：

- 空张量和单元素；
- `4096` 对齐 shape；
- `4219` 非对齐 shape，验证尾部 mask；
- FP16、FP32；
- dtype 不一致时明确报错。

2026-07-17 在当前 CUDA/Triton 环境中手动执行，结果为 `9 passed`。

BF16 已列入支持契约，但仍应在支持 BF16 的 GPU 上补一次动态测试。

## 5. Benchmark 应回答的问题

不能只记录“跑到了多少 GB/s”，还要记录：

1. GPU 型号、PyTorch/Triton/CUDA 版本和 dtype；
2. latency 中位数与分位数；
3. Triton 和 PyTorch 的 effective GB/s；
4. 小输入是否受 launch overhead 限制；
5. 大输入是否进入稳定带宽平台；
6. `BLOCK_SIZE=256/512/1024` 对性能是否有实际影响。

在没有硬件计数器证据前，不把 effective GB/s 直接表述成“100% DRAM 物理带宽利用率”。

## 6. Baseline 实测记录

测试环境：

```text
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
SM: 26
VRAM: 7.96 GiB
Compute Capability: 12.0
PyTorch: 2.11.0+cu128
Triton: 3.6.0
dtype: FP32
warmup/rep: 100/300 ms
```

曲线可以分成三个区域：

| 区域 | N | 现象 | 主要限制 |
| --- | ---: | ---: | --- |
| 固定开销区 | `4K ~ 64K` | 约 `6 ~ 64 GB/s` | launch、输出分配和计时粒度相对计算量过大 |
| 并行度爬升区 | `128K ~ 2M` | 约 `124 ~ 308 GB/s` | program 数量逐步增加，延迟隐藏能力提高 |
| 稳定平台区 | `4M ~ 64M` | 约 `328 ~ 346 GB/s` | 工作集足够大，进入当前环境的稳定有效带宽平台 |

在 `N=2^24` 的单点摘要中，Triton P50 为 `0.7585 ms`，即 `265.42 GB/s`；随后曲线中的同 shape 复测约为 `340 GB/s`，同时误差带明显变宽。这说明该点存在运行状态波动，可能与动态频率、Laptop 功耗状态、WSL/桌面调度或 cache 状态有关。

因此性能结论使用稳定区的多点趋势，而不是挑选单次最好值：

- 大规模下 Triton 与 PyTorch 曲线基本重合；
- 单点 P50 speedup 为 `1.003x`，应解释为性能相当，而不是 Triton 获得了有意义的领先；
- 当前算法已经正确表达连续读两次、连续写一次的 memory-bound 模式；
- 下一轮改变 `BLOCK_SIZE`，目的是研究 program 粒度和并行度，不预设它一定突破平台带宽。

## 7. BLOCK_SIZE Sweep

对 `BLOCK_SIZE=256/512/1024/2048` 进行相同 shape、相同计时参数的扫描后观察到：

- 小 shape 的结果主要受固定开销和计时波动影响，误差带大量重叠；
- 中等 shape 会出现局部差异，但没有某个 BLOCK_SIZE 在所有点上持续领先；
- `N >= 8M` 后四条曲线汇合到约 `335 ~ 346 GB/s`；
- 稳定平台区的差异基本不到 1%，远小于跨次运行的动态频率和功耗波动。

这说明当 grid 已经提供足够多的 program 后，继续改变单个 program 的 tile 长度不会突破全局内存通路的上限。

最终保留 `BLOCK_SIZE=1024` 作为默认值。这里不引入 autotune，因为它会增加多版本编译和缓存成本，却没有稳定、可复现的性能收益。

`01_vector_add` 至此完成闭环：正确性、输入契约、性能模型、baseline、参数控制实验和复盘均已完成。
