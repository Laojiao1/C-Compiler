# Softmax / Triton / Online Softmax 面试题

# 一、Softmax 数学基础

## 1. 为什么 Softmax 要减去最大值？

参考答案：直接计算大正数的 exp 会溢出。减去最大值后所有指数输入都不大于 0，最大为 `exp(0)=1`。Softmax 对整行平移不敏感，因此结果不变。

## 2. 证明 `softmax(x)=softmax(x-c)`。

参考答案：

\[
\frac{e^{x_i-c}}{\sum_j e^{x_j-c}}
=\frac{e^{x_i}e^{-c}}{e^{-c}\sum_j e^{x_j}}
=\frac{e^{x_i}}{\sum_j e^{x_j}}
\]

## 3. Softmax 为什么不是逐元素算子？

参考答案：每个输出都依赖整行分母 `sum(exp(x))`，Safe Softmax 还依赖整行最大值，因此包含两个全局 reduction。

## 4. 只检查输出行和为 1，能否证明 Softmax 正确？

参考答案：不能。错误的均匀分布也可能和为 1。行和只能作为辅助不变量，仍需与可靠 reference 逐元素比较。

## 5. Softmax 与 Cross Entropy 融合有什么好处？

参考答案：可以使用稳定的 log-sum-exp 表达，避免显式物化概率中间张量，并利用组合梯度 `p-y`，减少访存和数值风险。

# 二、Triton Fused Softmax

## 6. 当前 Fused Softmax 的并行划分是什么？

参考答案：grid 为 `(n_rows,)`，一个 Triton program instance 处理一行；行之间并行，行内通过 blocked tensor 做 max/sum reduction。

## 7. 为什么 BLOCK_SIZE 取 `next_power_of_2(n_cols)`？

参考答案：`tl.arange` 和 reduction 需要静态、规整的 blocked shape；2 的幂便于 lane 布局和二叉 reduction。多余 lane 通过 mask 保护。

## 8. masked load 为什么填 `-inf`，不能填 0？

参考答案：`-inf` 不会影响 max，且 `exp(-inf)=0` 不会进入分母。填 0 时，padding 可能成为最大值，并通过 exp 污染分母。

## 9. 为什么 load 和 store 都必须加 mask？

参考答案：load mask 阻止越界读，store mask 阻止越界写。无效 lane 的数值即使为 0，也不代表它对应的地址可以安全写入。

## 10. blocked tensor 是否等于 CUDA Shared Memory？

参考答案：不是。它是 Triton 的逻辑张量抽象，通常映射到寄存器等片上资源；只有编译器和具体 lowering 决定实际存储位置。tile 过大可能造成寄存器压力和 spill。

## 11. 为什么只要求 `stride(1)==1`，不要求整个 Tensor contiguous？

参考答案：Kernel 显式使用 row stride 跳行，但列地址直接 `row_start+cols`，因此行内必须连续；行间允许存在物理间隔。

## 12. Fused Softmax 的理想 FP32 算法流量是多少？

参考答案：输入读取一次 4MN，输出写入一次 4MN，总计 8MN Bytes。

# 三、Online Softmax

## 13. 为什么长行可能不适合整行 Fused Softmax？

参考答案：整行 blocked tensor 会增加寄存器需求和 reduction 规模，可能降低 occupancy、发生 spill、遇到编译或 tile 上限。

## 14. 朴素分块 Safe Softmax 为什么需要三趟输入读取？

参考答案：第一趟求 global max，第二趟使用最终 max 求 exp sum，第三趟使用最终 max/sum 输出概率。

## 15. Online Softmax 的状态不变量是什么？

参考答案：处理完当前前缀后：

\[
m_i=\max(\text{prefix})
\]

\[
l_i=\sum_{x\in prefix}e^{x-m_i}
\]

## 16. 推导 Online Softmax 的 l 更新公式。

参考答案：令 `m_new=max(m_i,m_chunk)`，利用：

\[
e^{x-m_{new}}=e^{x-m_i}e^{m_i-m_{new}}
\]

得到：

\[
l_{new}
=l_i e^{m_i-m_{new}}
+\sum_{chunk}e^{x-m_{new}}
\]

## 17. alpha 的范围是多少，为什么？

参考答案：`alpha=exp(m_i-m_new)`，由于 `m_new>=m_i`，所以指数不大于 0，`0<=alpha<=1`。旧贡献只会保持或缩小。

## 18. 为什么必须最后才执行 `m_i=m_new`？

参考答案：alpha 需要旧 `m_i`。若提前更新，`m_i-m_new=0`，alpha 永远为 1，历史贡献无法正确 rescale。

## 19. 第一趟已经得到全局 m/l，为什么仍需第二趟？

参考答案：m/l 只是两个归一化标量，不能恢复 N 个原始元素。要输出每个概率，必须重新读取对应 x。

## 20. Online Softmax 的空间复杂度如何准确描述？

参考答案：跨 chunk 持久状态 m/l 为 O(1)，当前工作 tile 为 O(BLOCK_SIZE)。BLOCK_SIZE 固定时，相对于 N 的辅助空间为 O(1)。

## 21. Online Softmax 的 FP32 算法流量是多少？

参考答案：两次输入读取 8MN，加一次输出写入 4MN，总计 12MN Bytes；每个元素还计算两次 exp。

# 四、GPU 性能分析

## 22. BLOCK_SIZE 是否越大越好？

参考答案：不是。大 BLOCK_SIZE 减少循环次数，但增加 reduction 和寄存器压力，可能降低 occupancy 或 spill；小 BLOCK_SIZE 资源压力低，但串行 chunk 和状态更新次数更多。

## 23. 为什么总元素数相同，M 小 N 大可能更慢？

参考答案：当前一个 program 处理一行，所以 program 数等于 M。M 小时可调度任务少，N 大又让每个 program 串行循环更多 chunk，SM 利用率和延迟隐藏能力下降。

## 24. 什么是 register spill？

参考答案：寄存器需求超过预算时，部分值被放入 Local Memory。Local Memory 是线程私有逻辑空间，物理访问可能经过 L1/L2 并最终到设备显存，延迟和流量代价高。

## 25. Effective GB/s 是否等于真实 DRAM 带宽？

参考答案：不等于。它用算法字节量除以时间；真实流量受 L1/L2、事务粒度、写策略和重复访问影响，需要 profiler 硬件计数器确认。

## 26. 为什么 Online Algorithmic GB/s 可能高于 Vector Add 平台？

参考答案：12MN 是模型字节量，第二趟读取可能命中 cache，并非所有模型字节都访问 DRAM，所以估算值可以高于基于流式访存观察到的平台。

## 27. Benchmark 为什么优先报告 latency？

参考答案：不同算法可能使用不同字节模型。latency 直接回答谁更快；GB/s 用来解释瓶颈，不能用不同分子掩盖完成同一任务所需时间。

## 28. 为什么要运行两次独立 benchmark？

参考答案：用于区分 shape 固有趋势与动态频率、功耗、WSL/桌面调度等运行噪声。只有可重复现象才适合进一步归因。

# 五、正确性测试

## 29. Online 测试为什么使用非 BLOCK_SIZE 整数倍列数？

参考答案：同时验证多 chunk 状态更新和最后一个 chunk 的 load/store mask，暴露越界与 padding 污染。

## 30. 为什么把较大值放在最后一个 chunk？

参考答案：让前面先积累历史 l，再在后段改变全局最大值，强制执行非平凡 alpha rescale。

## 31. 为什么测试使用 8 而不是 500？

参考答案：500 会让 alpha 下溢为 0，历史贡献数学上也几乎消失，错误地丢弃历史值可能仍通过。8 让 alpha 非零，历史贡献保持可观察。

## 32. 为什么测试失败不能只打印？

参考答案：pytest 依赖异常或断言判断失败。只打印错误但正常返回，测试仍可能显示 PASSED。

# 六、FlashAttention 衔接

## 33. 从 Online Softmax 到 FlashAttention 需要新增什么状态？

参考答案：新增输出 accumulator：

\[
acc=\sum_j e^{S_j-m}V_j
\]

最终输出 `O=acc/l`。

## 34. 最大值改变时为什么 acc 也要乘 alpha？

参考答案：旧 acc 基于旧最大值。分子和分母必须统一到同一个新指数基准，因此 l 和 acc 都乘 `exp(m_old-m_new)`。

## 35. 为什么融合 `softmax(S)@V` 后不需要第二趟？

参考答案：每个概率 chunk 的影响已经立即乘 V 并累加到 acc，不需要输出完整概率矩阵；循环结束直接 `acc/l`。

## 36. 当前 `kernel_FlashAtten.py` 为什么不是真正完整 FlashAttention？

参考答案：它接收已物化的分数矩阵 X，只融合 `softmax(X)@V`；真正 FlashAttention 要在片上由 Q/K tile 计算分数，并支持 scale、mask、batch/head、query tile 和 backward 等。

## 37. FlashAttention 主要减少计算量还是访存量？

参考答案：核心是减少 HBM I/O 和 O(N²) 中间矩阵物化，并没有消除 QKᵀ 和 PV 的主要矩阵乘计算。

# 七、开放设计题

## 38. M 很小、N 极大时，如何提高当前 Online Softmax 的并行度？

参考答案要点：让多个 program 处理同一行的不同区间，先产生局部 m/l，再通过额外阶段合并全局状态，最后并行输出。代价是额外 kernel、中间状态、同步和全局访存。

## 39. 如何选择 Online Softmax 的 BLOCK_SIZE？

参考答案要点：对代表性 M/N 做受控 sweep，同时记录 latency、寄存器、occupancy 和 spill；平衡 chunk 循环次数与单 tile 资源压力，不能只选单点最快值。

## 40. 如何证明 PyTorch 在某个 N 切换了 Kernel 路径？

参考答案要点：使用 profiler 记录 kernel name、grid/block、寄存器、共享内存和硬件计数器，并结合 PyTorch CUDA Softmax dispatch 源码；仅凭性能曲线只能提出假设。
