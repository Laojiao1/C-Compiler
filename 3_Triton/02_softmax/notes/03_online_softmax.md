# Online Softmax：数学、Kernel 与性能模型

## 1. 为什么需要分块

整行 Fused Softmax 要在一个 program 中表达完整行 tile。N 很大时，blocked tensor 会带来过高寄存器压力、低 occupancy、spill 或编译限制。

Online Softmax 固定每次只处理：

```text
BLOCK_SIZE 个元素
```

让单次工作集不随 N 无限增长。

## 2. 朴素分块为什么需要三趟

Safe Softmax 依赖全局最大值和全局分母：

```text
第一趟：读取 x，求 global max
第二趟：读取 x，求 exp sum
第三趟：读取 x，输出 probability
```

FP32 算法流量：

```text
三次读取 + 一次写入 = 16MN Bytes
```

Online Softmax 将 max 和 exp sum 合并到第一趟，把输入读取降到两次。

## 3. 核心状态不变量

处理完前 i 个 chunk 后，保存两个标量：

\[
m_i=\max(\text{已处理的所有元素})
\]

\[
l_i=\sum_{x\in\text{已处理前缀}}e^{x-m_i}
\]

`l_i` 不是当前 chunk 的和，也不是简单的 `sum(exp(x))`。它是所有历史元素统一到当前最大值基准后的稳定指数和。

## 4. 状态更新公式

当前 chunk 最大值：

\[
m_{ij}=\max(\text{chunk})
\]

新最大值：

\[
m_{new}=\max(m_i,m_{ij})
\]

旧状态的修正系数：

\[
\alpha=e^{m_i-m_{new}}
\]

根据指数拆分：

\[
e^{x-m_{new}}=e^{x-m_i}e^{m_i-m_{new}}
\]

得到：

\[
\boxed{
l_{new}=l_i\alpha+\sum_{x\in chunk}e^{x-m_{new}}
}
\]

代码映射：

```python
m_ij = tl.max(chunk, axis=0)
m_new = tl.maximum(m_i, m_ij)
alpha = tl.exp(m_i - m_new)
l_i = l_i * alpha + tl.sum(tl.exp(chunk - m_new), axis=0)
m_i = m_new
```

必须在最后才提交 `m_i=m_new`；如果提前更新，`alpha` 会错误地恒等于 1。

## 5. 初始状态为什么是 `-inf/0`

```python
m_i = -float("inf")
l_i = 0.0
```

第一块到来时：

```text
m_new = local max
alpha = exp(-inf - m_new) = 0
old contribution = 0
```

因此首个 chunk 自动退化成普通 Safe Softmax 状态，不需要额外分支。

## 6. padding 为什么仍然是 `-inf`

```python
chunk = tl.load(..., mask=mask, other=-float("inf"))
```

无效 lane：

```text
不会成为最大值
exp(-inf - m_new) = 0
不会进入 l_i
```

循环只在 `block_offset < n_cols` 时执行，因此最后一个 chunk 至少有一个有效元素，不会出现完全由 padding 组成的常规 chunk。

## 7. 为什么仍需第二趟

第一趟结束后：

```text
m_i = 整行最终最大值
l_i = 整行最终稳定指数和
```

但最终输出需要每个原始元素：

\[
p_j=\frac{e^{x_j-m_i}}{l_i}
\]

早期 chunk 的逐元素值已经丢弃，所以必须重新读取输入。

```python
chunk = tl.load(...)
probs = tl.exp(chunk - m_i) / l_i
tl.store(..., probs, mask=mask)
```

“Online”表示 `m/l` 可以在线更新，不表示最终概率第一次看到时就能永久输出。

## 8. 空间复杂度的准确表述

跨 chunk 持久状态：

```text
m_i + l_i = O(1)
```

当前工作 tile：

```text
O(BLOCK_SIZE)
```

当 BLOCK_SIZE 固定时，相对于 N 的辅助空间为 O(1)。输出张量 O(MN) 通常不计入额外辅助空间。

## 9. GPU 并行映射

```python
grid = (n_rows,)
```

所以：

```text
program 数量 = M
每个 program 负责一行
行之间并行
同一行的 chunk 顺序执行
chunk 内 BLOCK_SIZE 个 lane 并行
```

每个 program 的循环次数：

\[
\left\lceil\frac{N}{BLOCK\_SIZE}\right\rceil
\]

性能由三者共同决定：

```text
M：跨行 program 并行度
N/BLOCK_SIZE：单行串行循环长度
BLOCK_SIZE：单个 tile 的片上资源需求
```

## 10. BLOCK_SIZE 权衡

### 更小 BLOCK_SIZE

```text
优点：tile 小，寄存器压力低
缺点：循环次数多，m/l 依赖链长
```

### 更大 BLOCK_SIZE

```text
优点：循环次数少
缺点：reduction 大，寄存器压力高，occupancy 可能下降，严重时 spill
```

register spill 会落入 Local Memory；Local Memory 是线程私有逻辑空间，物理访问可能经过 L1/L2 并最终到设备显存。

## 11. 输入契约

Python 封装要求：

- CUDA 二维 Tensor；
- dtype 为 FP16/BF16/FP32；
- `stride(1)==1`；
- BLOCK_SIZE 是不超过上限的正 2 的幂；
- 空张量直接返回。

行间 stride 可以大于 n_cols，因为 Kernel 显式使用 `input_row_stride`。

## 12. 访存与计算代价

FP32 Online Softmax：

```text
第一趟读取 x：4MN
第二趟读取 x：4MN
写入 y：       4MN
总计：        12MN Bytes
```

相比 Fused 的 8MN，多 50% 算法流量；同时每个元素需要计算两次 exp。

它的价值是资源可控和长行支持，不是保证常规 shape 更快。

## 13. 当前 benchmark 的核心结果

固定总元素 `M*N=2^24`：

```text
M=4096, N=4096, chunks=4:
Online P50 = 0.4446 ms

M=16, N=1048576, chunks=1024:
Online P50 = 1.4280 ms
```

总元素相同，少行长列约慢 3.21 倍，证明任务形状和 program 并行度非常重要。

固定 `M=256` 时，Online 在多数长行上比 PyTorch 慢约 20%~25%，符合额外读取、额外 exp 和串行 chunk 的代价。个别 shape 反超 PyTorch，可能与 PyTorch dispatch 或运行波动有关，不能推广成普遍结论。

## 14. 当前实现的定位

这是一个教学型、单 program 单行的 Online Softmax：

- 数学正确；
- 工作 tile 固定；
- 能处理长行；
- 没有跨 program 合并；
- M 很小时可能无法填满所有 SM；
- 尚未做 BLOCK_SIZE/num_warps 系统调优；
- 不是所有 shape 的工业最优实现。

## 15. 核心记忆

- Online Softmax 的核心不是分块，而是维护 m/l 状态不变量
- alpha 用于把历史贡献换到新最大值基准。
- 第一趟得到全局 m/l，第二趟生成逐元素概率。
- 它把朴素三趟降为两趟，但无法达到整行 Fused 的一次读取。
