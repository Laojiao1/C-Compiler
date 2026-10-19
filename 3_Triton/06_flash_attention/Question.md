# FlashAttention Forward 面试问答

本文用于复习和模拟面试。回答时不必逐字背诵，重点记住每个结论背后的因果关系。

## 1. 请介绍一下标准 Attention 的计算过程

**参考回答：**

对每个 batch 和 head，输入 `Q/K/V` 的形状都是 `[N,D]`。Attention Forward 分为三步：

\[
S=\frac{QK^T}{\sqrt D}
\]

\[
P=\operatorname{softmax}(S)
\]

\[
O=PV
\]

`S/P` 的形状是 `[N,N]`，输出 `O` 的形状是 `[N,D]`。缩放因子 `1/sqrt(D)` 用于避免 `D` 较大时点积数值过大，使 Softmax 过早饱和。

**常见追问：为什么 Softmax 是逐行计算？**

因为每个 Query 行都要在所有 Key 位置之间独立分配概率，所以每行概率之和为 1。

## 2. 为什么 Naive Attention 要保存完整的 S 和 P？

**参考回答：**

在本项目的 Naive 实现中，计算被拆成三个 Kernel：

```text
Score Kernel -> Softmax Kernel -> Output Kernel
```

后一个 Kernel 无法直接使用前一个 Kernel 的寄存器或片上临时变量，因此：

- Score Kernel 必须把完整 S 写到全局显存；
- Softmax Kernel 读取 S，再把完整 P 写到全局显存；
- Output Kernel 读取 P 和 V，计算 O。

所以 S 和 P 不仅占用 `O(BHN²)` 容量，还会产生额外的全局显存写入和读取。

例如 `B=2、H=8、N=1024、FP16` 时，单个矩阵占 32 MiB，S+P 占 64 MiB。只算各写一次、读一次，也至少有约 128 MiB 的中间显存流量。

**常见追问：数学上是否必须保存 S/P？**

不是。数学公式只要求得到相同结果。Naive 多 Kernel 执行方式需要保存，而 FlashAttention 通过融合和在线 Softmax 避免保存完整 S/P。

## 3. FlashAttention 的核心思想是什么？

**参考回答：**

FlashAttention 不改变 Attention 的数学结果，而是改变计算顺序和中间数据的生命周期。

它让一个 Program 固定一个 Query Tile，在内部依次遍历多个 K/V Tile。每次只计算一个小 Score Tile，然后立即更新在线 Softmax 状态和输出累加器。当前 Tile 使用结束后直接丢弃，最终只把 O 写回全局显存。

核心可以概括为三点：

1. 分块计算，不保存完整 S/P；
2. 使用在线 Softmax 正确合并多个 Key Tile；
3. 融合 `QK^T`、缩放、遮罩、Softmax 和 `P@V`，减少显存往返。

**常见追问：FlashAttention 是否不再计算 S 和 P？**

不是。它仍然计算当前 Tile 的 Score 和 Softmax 权重，只是不把完整的 N×N 中间矩阵写回显存。

## 4. 为什么不能分别对每个 Key Tile 做普通 Softmax？

**参考回答：**

因为 Softmax 的分母必须包含该 Query 行的全部 Key：

\[
p_j=\frac{e^{s_j}}{\sum_t e^{s_t}}
\]

如果每个 Key Tile 分别做 Softmax，每一块的概率之和都是 1。两个块直接拼接后概率总和会变成 2，而不是 1。

不同块还可能有不同最大值，局部概率使用的指数基准和分母都不同，所以不能直接相加。必须保存可合并的 `m、l、acc` 状态。

**常见追问：能不能最后再把局部概率整体除以块数？**

不能。不同块的 Score 分布不同，每块在全局 Softmax 中的真实权重并不相等，简单除以块数会得到错误结果。

## 5. 在线 Softmax 中的 m、l、acc 分别是什么？

**参考回答：**

对于一个 Query 行，处理到当前 Key 位置时：

\[
m=\max(s_1,\ldots,s_t)
\]

`m` 是目前见过的运行最大值，用于稳定指数计算。

\[
l=\sum_{j=1}^{t}e^{s_j-m}
\]

`l` 是以当前 m 为基准的未归一化权重总和。

\[
acc=\sum_{j=1}^{t}e^{s_j-m}V_j
\]

`acc` 是使用同一组未归一化权重得到的 Value 加权和。

所有 Key Tile 处理完后：

\[
O=\frac{acc}{l}
\]

分子和分母使用相同的指数基准，所以相除后就是 `softmax(S)@V`。

**常见追问：为什么不在每轮都计算 O？**

可以得到暂时的局部 O，但后续块可能改变最大值和分母。维护未归一化的 l 和 acc 更容易进行精确合并，最后只需除一次。

## 6. 新 Key Tile 出现更大最大值时，为什么要重新缩放？

**参考回答：**

旧状态以 `m_old` 为指数基准，新状态要统一到：

\[
m_{new}=\max(m_{old},m_{block})
\]

旧权重需要乘：

\[
\alpha=e^{m_{old}-m_{new}}
\]

因为：

\[
e^{s-m_{old}}\alpha=e^{s-m_{new}}
\]

更新公式为：

\[
l_{new}=\alpha l_{old}+\sum e^{s-m_{new}}
\]

\[
acc_{new}=\alpha acc_{old}+\sum e^{s-m_{new}}V
\]

这相当于先把旧结果转换到新的统一标尺，再与新块合并。

**常见追问：为什么 l 和 acc 必须同时缩放？**

因为 l 和 acc 必须代表同一组权重。如果只缩放 l，旧 Value 在 acc 中仍保留原来的较大权重，最终 `acc/l` 会偏离真实结果。

## 7. 为什么最终输出是 O=acc/l？

**参考回答：**

`acc` 保存的是未归一化加权和：

\[
acc=\sum_j e^{s_j-m}V_j
\]

`l` 保存对应权重的总和：

\[
l=\sum_j e^{s_j-m}
\]

所以：

\[
\frac{acc}{l}
=\sum_j\frac{e^{s_j-m}}{\sum_t e^{s_t-m}}V_j
=\sum_j\operatorname{softmax}(s)_jV_j
\]

减去 m 只改变分子和分母的共同倍数，不改变最终概率。

**常见追问：m 会不会改变 Attention 结果？**

不会。分子和分母同时乘了 `e^{-m}`，相除后抵消。它只用于数值稳定。

## 8. 一个 Triton Program 具体负责什么？

**参考回答：**

当前输入形状是 `[B,H,N,D]`，正式配置是：

```text
BLOCK_M=64
BLOCK_N=32
num_warps=4
```

Grid 为：

\[
(\lceil N/BLOCK\_M\rceil, B\times H)
\]

一个 Program 固定负责一个 batch、一个 head 和最多 64 个连续 Query 行。它在内部循环遍历这些 Query 需要的全部 K/V Tile，持续更新 m、l、acc，最后写出这些 Query 行对应的 O。

每轮主要 Shape 是：

```text
Q Tile      [64,D]
K^T Tile    [D,32]
V Tile      [32,D]
Score Tile  [64,32]
acc         [64,D]
m、l        [64]
```

**常见追问：为什么 Key Tile 不放进 Grid？**

如果不同 Key Tile 由不同 Program 处理，它们无法直接共享同一个 Query Tile 的 m、l、acc，还需要跨 Program 合并和同步。当前设计让在线状态始终留在一个 Program 内。

## 9. Query Tile、Key Tile 和 Value Tile 如何复用数据？

**参考回答：**

一个 Program 先加载 Q Tile，然后跨全部 Key Tile 循环复用它。每轮加载一块 K 和对应的 V：

```text
Q Tile @ K^T Tile -> Score Tile
Score Tile -> 在线 Softmax 权重
权重 @ V Tile -> 更新 acc
```

当前 Score 和权重 Tile 用完即丢弃，K/V Tile 进入下一轮时也被替换。跨循环长期保留的是 Q Tile 和 m、l、acc。

**常见追问：这些变量一定都在寄存器里吗？**

不能只看源码就绝对确认。可以说它们是 Program 内的片上候选数据，具体寄存器数量、共享内存使用和是否 spill 需要检查编译结果或 Profiler。

## 10. 因果遮罩在分块中如何处理？

**参考回答：**

因果 Attention 要求 Query 位置 i 只能访问：

\[
j\le i
\]

分块后，一个 Key Tile 对当前 Query Tile 可能是：

- 完全可见；
- 部分可见，需要逐元素判断 `j<=i`；
- 完全位于未来，可以跳过整个 Tile。

未来 Score 被设为负无穷，随后对应权重显式置零，所以不会影响 m、l 和 acc。

当前正式入口在 `causal=True 且 N>=512` 时跳过完全未来的 Key Tile。

**常见追问：既然实现了因果跳块，为什么部分可见的 Key Tile 仍然不能整个跳过？**

> 因果跳块只能跳过对当前 Query Tile 中所有 Query 行都不可见的 Key Tile。部分可见 Tile 同时包含合法位置和未来位置，如果整个跳过会丢失合法贡献。因此当前实现仍然计算该 Tile，再使用 `j<=i` 的逐元素 mask 将未来位置权重置零。只有完全位于未来的 Tile 才不会进入计算循环。

## 11. 为什么同时测试因果和非因果？

**参考回答：**

两者经过不同的控制流和有效数据范围。

非因果模式会访问全部 Key，适合验证完整分块和在线 Softmax。因果模式还包含逐元素三角 mask、部分可见 Tile 和未来 Tile 跳过，边界更复杂。

如果只测因果，无法确认无 mask 的完整路径是否正确，也无法分辨性能收益来自 Flash 融合还是来自少算了未来 Tile。

**常见追问：为什么因果模式有时反而更快？**

当前长序列实现会跳过接近一半完全未来的 Tile，因此执行的 Tile 计算更少。但复杂 mask 也有开销，所以小序列不一定更快。

## 12. 非对齐的 N 和 D 如何保护？

**参考回答：**

最后一个 Tile 可能超出真实形状，所以分别建立：

```text
query_is_valid
key_is_valid
feature_is_valid
```

无效 Q/K/V 位置不会参与有效计算；无效 Score 被视为负无穷，并把对应权重显式置零；最后只写回有效 Query 行和特征。

`BLOCK_D` 会补齐到不小于 16 的二次幂。例如 `D=40` 使用 `BLOCK_D=64`，多出的 24 个特征通过 mask 保护。

**常见追问：为什么不能只在 store 时做 mask？**

因为越界 load 本身就可能非法，而且无效 Score 如果进入 max、l 或 acc，也会污染有效输出。

## 13. 为什么 m、l、acc 使用 FP32？

**参考回答：**

它们会跨多个 Key Tile 累积。FP16/BF16 的有效精度更低，如果直接保存这些运行状态，容易放大舍入误差，较小指数还可能下溢。

因此当前实现用 FP32 保存运行最大值、分母和输出累加器，最终 `acc/l` 写回时再转换为输入 dtype。

这是常见的混合精度策略：输入和矩阵乘可以利用低精度吞吐，关键归约和累加使用 FP32 保证稳定性。

**常见追问：FP32 累加能保证结果完全一致吗？**

不能保证逐 bit 一致。分块顺序、矩阵乘精度和浮点加法顺序都可能不同，所以 correctness 使用合理的 `atol/rtol` 判断数值接近。

## 14. FlashAttention 是否把复杂度从 O(N²) 降到了 O(N)？

**参考回答：**

没有。

`QK^T` 和 `P@V` 仍然需要处理约 N² 个 Query/Key 组合，计算复杂度仍是：

\[
O(BHN^2D)
\]

FlashAttention 减少的是完整 N×N 中间张量的显存容量和全局显存流量，而不是消除这些点积计算。

因果跳块大约可以减少一半 Tile 工作量，但常数减半后仍然是 `O(N²)`。

**常见追问：为什么没有降低渐近复杂度却能快很多？**

GPU 性能不仅由 FLOPs 决定。Naive 实现需要频繁读写大规模 S/P，Flash 通过融合和数据复用减少全局显存访问，所以相同数量级的计算可以更快完成。

## 15. Tile 是否越大越快？

**参考回答：**

不一定。

更大的 Tile 能增加数据复用、减少 Program 数和循环次数，但也会扩大 Score Tile 和 `acc[BLOCK_M,D]`，增加寄存器或片上资源压力，可能降低 occupancy，甚至产生 spill。

更多 warps 也不一定更快，因为它可能提高并行度，也可能增加资源占用和调度成本。

因此 Tile 和 warp 必须在目标 Shape 和硬件上通过控制变量实验选择。

**常见追问：本项目最终为什么选择 64×32、4 warps？**

因为它在不同 N、因果和非因果场景中表现较稳定。`64×64` 只在少数点略好，8 warps 多数较差，不足以支持更复杂的 Shape heuristic。

## 16. 你的 correctness 是怎样分层验证的？

**参考回答：**

我先用 05 的 PyTorch 显式 Attention 作为 reference，然后分层覆盖：

- 对齐和非对齐 N/D；
- 极小序列、空序列和多个 Key Tile；
- FP16、BF16、FP32；
- 因果和非因果；
- 不同 batch/head；
- 与 Triton Naive 交叉验证；
- 所有 Tile/warp 调优候选；
- 因果跳块路径；
- 非法 Tensor、Shape、dtype、连续性、D、Tile 和参数。

本轮共有 77 项通过。不支持 BF16 的设备预期为 69 项通过、8 项跳过。

正式 Benchmark 在每个 Shape 计时前还会再次比较输出，防止只测到了一个很快但错误的 Kernel。

**常见追问：为什么性能候选也必须做 correctness？**

因为不同 Tile 会改变 mask、循环次数和边界路径。某个配置能编译和运行，不代表它在非对齐或因果场景仍然正确。

## 17. Benchmark 如何保证比较公平？

**参考回答：**

三种实现使用相同的 Q/K/V、Shape、dtype 和因果设置：

- PyTorch 显式 Naive；
- 05 Triton Naive；
- 06 Triton Flash。

输入创建、correctness、JIT 和预热都在正式计时前完成。使用 Triton `do_bench` 记录 P20/P50/P80，而不是只看一次运行时间。

正式 Shape 固定为 `B=2,H=8,D=64,FP16`，测试 `N=128～2048` 的因果和非因果场景。

PyTorch SDPA 没有放进主要 Naive 对照，因为它可能自动选择融合后端，不再代表显式保存 S/P 的相同算法路径。

**常见追问：为什么要记录 P20/P50/P80？**

P50 表示典型延迟，P20～P80 可以观察波动。如果两个方案 P50 很接近，但分位区间大量重叠，就不应该轻易宣布某个方案稳定更快。

## 18. 峰值显存是怎样测量的？

**参考回答：**

测量前 Q/K/V 已经存在，然后重置 CUDA 峰值统计，记录一次 provider 调用相对 baseline 增加的活跃显存峰值。

所以结果主要反映输出 O、S/P 和临时张量，不包含已经存在的 Q/K/V，也不等于 CUDA 缓存分配器保留的全部显存。

在 `N=2048` 时，Triton Naive 的峰值增量约为 260 MiB，Flash 约为 4 MiB。Flash 的结果基本等于输出 O 的容量，支持它没有显式保存完整 S/P。

**常见追问：能否说 Flash 总共只使用 4 MiB 显存？**

不能。4 MiB 是相对已存在输入的峰值增量，不包含 Q/K/V、框架上下文、缓存分配器和其他运行时显存。

## 19. TFLOPS 是如何计算的？因果模式有什么特殊之处？

**参考回答：**

两个矩阵乘的完整等效计算量约为：

\[
FLOPs=4BHN^2D
\]

\[
TFLOPS=\frac{FLOPs}{latency_{ms}}\times10^{-9}
\]

正式 CSV 的因果模式仍使用完整 N² 计算量，所以得到的是“完整问题等效 TFLOPS”。因果跳块后，Kernel 实际没有执行全部 Tile，因此这个数不能直接当作硬件真实指令吞吐。

例如因果 `N=2048` 的完整等效结果是 40.84 TFLOPS，但实际只遍历约 51.56% 的 Tile，按 Tile 工作量估算约为 21.06 TFLOPS。

**常见追问：哪个数更适合表示性能？**

完整等效 TFLOPS 适合表示解决整个问题的速度；执行 Tile TFLOPS 更接近实际完成的计算工作；精确硬件吞吐仍需要 Profiler。

## 20. 哪些结论由实验直接支持？

**参考回答：**

当前实验直接支持：

- Flash 输出与 reference 一致；
- Flash 不显式保存完整 S/P；
- Flash 峰值增量显存不再包含 N² 的 S/P；
- 长序列相对 PyTorch Naive 和 Triton Naive 有明显优势；
- `64×32、4 warps` 是当前范围内较稳定的统一配置；
- 因果长序列跳过未来 Key Tile 有稳定收益。

例如 `N=2048` 时：

- 非因果 Flash 对 Triton Naive 加速约 4.83 倍；
- 因果 Flash 对 Triton Naive 加速约 7.54 倍；
- Triton Naive 峰值约 260 MiB，Flash 峰值约 4 MiB。

**常见追问：这是否证明了提升一定来自显存带宽？**

它强烈支持“减少中间张量和显存流量有帮助”，但不能只凭端到端延迟确定具体硬件瓶颈，还需要 Profiler 指标。

## 21. 哪些性能解释仍需要 Profiler 验证？

**参考回答：**

下面这些不能只靠延迟曲线确定：

- registers per thread；
- occupancy；
- 是否发生 register spill；
- L2 和显存带宽利用率；
- Tensor Core 利用率；
- warp stall reason；
- 具体 load/store 和 MMA 指令；
- 不同 Tile 的资源占用差异。

严谨的表达应该是：实验现象符合某个解释，而不是已经证明某个底层事件发生。

**常见追问：准备如何验证？**

可以使用 Nsight Compute 查看寄存器、occupancy、DRAM/L2 throughput、warp stall 和 Tensor Core 指标，再结合 TTIR、TTGIR、PTX/SASS 检查编译器最终生成的访存和矩阵乘指令。

## 22. 因果跳块实验得到了什么结论？

**参考回答：**

我固定 `64×32、4 warps`，只改变是否遍历完全未来的 Key Tile，并用三轮随机顺序做配对比较。

结果是：

```text
N=128    基本持平
N=256    跳块可能变慢
N=512    约 1.76x
N=1024   约 1.78x
N=2048   约 1.74x
```

因此正式策略为：

```text
非因果：不使用因果跳块
因果且 N<512：遍历全部 Key Tile
因果且 N>=512：跳过完全未来的 Key Tile
```

`512` 是当前设备和 Shape 的实验阈值，不是可以无条件迁移到所有 GPU 的固定规则。

**常见追问：为什么 N=256 少算了 Tile却可能更慢？**

小序列本身运行时间很短，节省的计算有限，控制流、调度、GPU 状态和测量波动都可能抵消收益。

## 23. 从初始版本到最终版本，你是怎样做性能决策的？

**参考回答：**

我的流程是：

```text
PyTorch reference
-> 最小 fused Flash Forward
-> 分层 correctness
-> 正式 Benchmark baseline
-> Tile/warp 控制变量实验
-> 三轮随机顺序复测
-> 选择统一配置
-> 因果跳块独立实验
-> 接入正式入口
-> 重新运行 correctness 和 Benchmark
```

最初 baseline 是 `32×32、4 warps`。第一轮实验发现 `64×32、4 warps` 整体较好，随后通过三轮随机顺序复测排除部分单轮偶然值。

`64×64` 在少数点更快，但没有形成跨 Shape 的稳定优势，所以没有加入复杂 heuristic。最后再单独验证因果跳块，并只把有稳定收益的范围接入正式入口。

**常见追问：为什么不直接保留每个 Shape 的最快配置？**

因为当前 Shape 覆盖和重复次数还不足以支持可靠的多分支规则。过早加入 heuristic 容易过拟合当前设备和少数测试点。

## 24. 哪些实验或直觉最终被推翻了？

**参考回答：**

主要包括：

1. 每个 Key Tile 分别做 Softmax再相加是错误的；
2. Tile 越大不一定越快；
3. warp 越多不一定越快，8 warps 多数较差；
4. 单轮最快不等于稳定最快；
5. 因果跳块不是所有 N 都有收益；
6. Flash 不保证在每个小 Shape 都比 Naive 快。

这些失败实验很重要，因为它们把最终选择从“凭直觉”变成了“由控制变量实验支持”。

**常见追问：如何对待失败优化？**

保留实验问题、控制变量、预测和结果，说明它为什么没有泛化。失败结果可以防止以后重复尝试同一个错误 heuristic。

## 25. 如何区分算法正确、工程可用和性能优秀？

**参考回答：**

算法正确表示输出与数学 reference 在合理误差范围内一致。

工程可用还要求接口约束清楚，错误输入能报错，空序列、非对齐 Shape、不同 dtype 和边界条件都有处理。

性能优秀则要求它在目标硬件和目标工作负载上稳定领先，并且能够解释收益来源和新的瓶颈。

当前版本已经是正确、可运行、可测量的 FlashAttention Forward，并在目标长序列上有明显性能优势，但还不是跨硬件、跨 Shape 自动调优的生产级实现。

**常见追问：为什么 correctness 通过不代表可以上线？**

因为生产环境还需要更完整的 Shape、设备、并发、稳定性、错误处理、维护成本和性能回归验证。

## 26. 当前实现支持什么？有什么限制？

**参考回答：**

当前支持：

- CUDA Tensor；
- Q/K/V 同为 `[B,H,N,D]`；
- FP16、BF16、FP32；
- 最后一维连续；
- `0<=N`、`1<=D<=128`；
- 因果和非因果 self-attention；
- 输出 Shape 和 dtype 与 Q 相同。

当前不支持：

- backward 和训练反向传播；
- Dropout、Attention bias、自定义 scale；
- Q 和 K/V 长度不同的 cross-attention；
- `D>128`；
- 返回完整 S/P；
- FlashAttention 2/3 的复杂优化；
- 跨设备自动选择 Tile 和跳块阈值。

**常见追问：下一步最值得补什么？**

按照当前学习路线，不继续扩展 backward，而是进入 Triton lowering 和 CUDA C++，通过 IR、PTX/SASS 与 Profiler 理解当前性能现象的底层原因。

## 27. 请用一分钟总结这个项目

**参考回答：**

> 我先实现了显式保存 Score 和 Probability 的 Triton Naive Attention，然后实现了融合的 FlashAttention Forward。
>
> Naive 多 Kernel 版本需要通过全局显存传递 `[B,H,N,N]` 的 S 和 P，所以中间容量与显存流量随 N² 增长。
>
> Flash 版本让一个 Program 固定一个 Query Tile，在内部遍历 K/V Tile，并使用 FP32 的运行最大值 m、分母 l 和输出累加器 acc 做在线 Softmax。
>
> 新块出现更大最大值时，旧 l 和 acc 同时乘 `exp(m_old-m_new)`，从而在不保存完整概率矩阵的情况下得到与全局 Softmax 相同的结果。
>
> 经过分层 correctness、Tile/warp 控制变量实验、三轮随机复测和因果跳块实验，我最终选择 `64×32、4 warps`，并在因果 `N>=512` 时跳过完全未来的 Key Tile。长序列取得了明显的延迟和显存优势，但寄存器、occupancy、带宽以及 Tensor Core 利用率仍需要 Profiler 验证。

## 28. 最后应记住的回答框架

遇到没有准备过的性能问题，可以按下面顺序回答：

```text
先说明数学语义
-> 再说明 Program/Tile 如何映射
-> 指出数据在哪里产生、保存和销毁
-> 说明理论上减少了什么或增加了什么
-> 给出 correctness 与 Benchmark 证据
-> 最后说明证据边界和需要的 Profiler 指标
```

这个框架比直接背“共享内存、occupancy、Tensor Core”更可靠，因为它能把算法、实现、实验和硬件证据连接起来。
