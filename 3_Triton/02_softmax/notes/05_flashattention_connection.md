# 从 Online Softmax 到 FlashAttention

对应教学代码：`../softmax_FlashAtten/kernel_FlashAtten.py`。

## 1. 普通 Attention 的中间矩阵

\[
S=QK^T
\]

\[
P=\operatorname{softmax}(S)
\]

\[
O=PV
\]

朴素实现会物化完整 S 和 P。序列长度为 N 时，两者都是 O(N²) 中间矩阵，并产生大量 Global Memory 写入和读取。

## 2. 普通 Online Softmax 为什么仍需第二趟

第一趟只保存全局：

```text
m：最大值
l：稳定指数和
```

若目标是输出完整概率 P，每个元素仍需要原始分数 `S_j`，因此必须第二次读取。

## 3. Attention 不需要单独输出 P

\[
O=\operatorname{softmax}(S)V
\]

代入 Softmax：

\[
O=
\frac{\sum_j e^{S_j-m}V_j}
{\sum_j e^{S_j-m}}
\]

定义：

\[
l=\sum_j e^{S_j-m}
\]

\[
acc=\sum_j e^{S_j-m}V_j
\]

最终：

\[
\boxed{O=acc/l}
\]

因此可以在看到每个分数 chunk 时，立即把它对 V 的贡献累加到 acc，不需要保存完整概率。

## 4. 三个在线状态

```python
m_i = -float("inf")
l_i = 0.0
acc = tl.zeros([D_V], dtype=tl.float32)
```

含义：

```text
m_i：已处理分数的最大值
l_i：已处理分数在 m_i 基准下的指数和
acc：已处理分数在 m_i 基准下的加权 V 累加结果
```

## 5. 更新公式

当前分数 chunk 的新最大值：

\[
m_{new}=\max(m_i,m_{chunk})
\]

修正系数：

\[
\alpha=e^{m_i-m_{new}}
\]

当前权重：

\[
p_{chunk}=e^{S_{chunk}-m_{new}}
\]

更新分母：

\[
l_{new}=l_i\alpha+\sum p_{chunk}
\]

更新输出累加器：

\[
acc_{new}=acc_i\alpha+\sum p_jV_j
\]

最后：

\[
O=acc/l
\]

## 6. 为什么 acc 也必须乘 alpha

旧 acc 基于旧最大值：

\[
acc_i=\sum_{old}e^{S_j-m_i}V_j
\]

新状态必须基于 `m_new`：

\[
e^{S_j-m_{new}}
=e^{S_j-m_i}e^{m_i-m_{new}}
\]

所以旧分子贡献也必须乘同一个 alpha。只修正 l、不修正 acc，会让分子分母处于不同指数基准，输出必错。

## 7. V tile 的地址广播

```python
v_ptrs = (
    V_ptr
    + cols[:, None] * stride_vk
    + vd_offs[None, :] * stride_vd
)
```

shape 变化：

```text
cols[:, None]      -> [BLOCK_SIZE, 1]
vd_offs[None, :]   -> [1, D_V]
广播后              -> [BLOCK_SIZE, D_V]
```

每个分数权重对应 V 的一行，再沿 BLOCK_SIZE 维求和，得到长度 D_V 的 chunk 输出贡献。

## 8. 为什么需要 d_mask

`tl.arange` 使用补齐到 2 的幂的 `D_V`：

```python
D_V_PAD = triton.next_power_of_2(d_v)
```

真实张量仍只有 d_v 列，因此必须：

```python
d_mask = vd_offs < d_v
```

并同时保护：

- V 的 load；
- O 的 store。

否则 `d_v=63/65` 等非 2 的幂会发生越界读写。当前测试专门覆盖 63、64、65。

## 9. 为什么融合后不需要第二趟

每个 chunk 的概率贡献已经被消费：

```text
读取分数 chunk
读取 V chunk
更新 m/l
更新 acc
丢弃当前 chunk
```

循环结束时，所有概率对最终 O 的影响都已进入 acc。最终只需 `acc/l`，不用输出或重新读取概率矩阵。

## 10. 当前代码不是真正完整 FlashAttention

当前接口接收：

```text
x = 已经物化的分数矩阵 QK^T
v = V
```

它只融合：

```text
softmax(X) @ V
```

真正 FlashAttention 需要在片上计算分数 tile：

```text
读取 Q tile
循环读取 K/V tile
计算 QK^T tile
应用 scale/mask
在线更新 m/l/acc
写出 O tile
```

从而既不物化完整 `QK^T`，也不物化完整概率矩阵。

## 11. 当前教学实现缺少什么

- batch 和 multi-head；
- `1/sqrt(d)` 缩放；
- causal mask；
- 真正的 Q/K dot-product tile；
- query tile，而不是单个分数行；
- K/V 在多个 query 行间的复用；
- num_warps/num_stages 调优；
- backward；
- 工业级 dtype/layout 支持。

## 12. 核心数据流对比

### 普通 Attention

```text
QK^T -> 写 S -> 读 S -> 写 P -> 读 P -> P@V
```

### FlashAttention 思想

```text
Q/K tile -> 局部 S -> online m/l/acc -> O
```

关键不是“少算了矩阵乘法”，而是避免把 O(N²) 中间矩阵反复写入和读取 HBM。

## 13. 核心记忆

- Online Softmax 保存 m/l。
- 融合 softmax(X)@V 还要保存 acc。
- 最大值改变时，l 和 acc 必须用同一个 alpha 修正。
- 真正 FlashAttention 进一步在片上生成 QK^T tile。
- 当前代码是数学桥梁，不是完整实现。
