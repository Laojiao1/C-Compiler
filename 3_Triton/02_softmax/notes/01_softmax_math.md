# Softmax 数学基础与数值稳定性

## 1. Softmax 解决什么问题

给定一行实数：

\[
x=[x_0,x_1,\ldots,x_{N-1}]
\]

Softmax 将它转换成概率分布：

\[
p_i=\frac{e^{x_i}}{\sum_j e^{x_j}}
\]

输出满足：

```text
p_i > 0
sum(p_i) = 1
```

它不是逐元素算子，因为每个输出都依赖整行分母。

## 2. 为什么直接计算会溢出

浮点数能表示的范围有限。若直接计算 `exp(1000)`，FP32 会得到 `inf`，随后可能出现：

```text
inf / inf -> NaN
```

Softmax 需要先减去全局最大值：

\[
m=\max_i x_i
\]

\[
p_i=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}
\]

因为：

\[
x_i-m\leq 0
\]

所以最大指数为 `exp(0)=1`，避免正向指数溢出。

## 3. 减去最大值为什么不改变结果

对任意常数 c：

\[
\frac{e^{x_i-c}}{\sum_j e^{x_j-c}}
=
\frac{e^{x_i}e^{-c}}{e^{-c}\sum_j e^{x_j}}
=
\frac{e^{x_i}}{\sum_j e^{x_j}}
\]

Softmax 对整行平移不敏感：

```text
softmax(x) = softmax(x - c)
```

选择 `c=max(x)`，是为了获得最稳定的指数范围。

## 4. Safe Softmax 的计算步骤

```py
m = max(x)
numerator = exp(x - m)
denominator = sum(numerator)
output = numerator / denominator
```

对应复杂度：

```text
计算量：O(N)
输出空间：O(N)
全局依赖：max reduction + sum reduction
```

## 5. Softmax 的两个重要不变量

### 平移不变量

\[
\operatorname{softmax}(x+c)=\operatorname{softmax}(x)
\]

### 概率和不变量

\[
\sum_i p_i=1
\]

测试时，概率和可以作为辅助检查，但不能替代与正确 reference 的逐元素比较。

## 6. 特殊值语义

### 输入包含一个 `+inf`

直接执行 `inf - inf` 会产生 `NaN`。不同框架可能对特殊值有专门处理，因此算子库必须明确自己与 PyTorch 的语义是否一致。

### 整行都是 `-inf`

最大值也是 `-inf`，计算 `-inf - (-inf)` 会得到 `NaN`。数学上这不是一个正常的有限概率分布输入。

### 输入包含 NaN

NaN 通常会通过 max、exp、sum 传播到输出。是否允许、传播方式是否与 reference 一致，都应通过测试固定下来。

当前学习代码主要验证有限浮点输入，NaN/Inf 语义属于待补测试。

## 7. Softmax 梯度补充

Softmax Jacobian：

\[
\frac{\partial p_i}{\partial x_j}
=p_i(\delta_{ij}-p_j)
\]

当 Softmax 与交叉熵组合时，logits 梯度可简化为：

\[
\frac{\partial L}{\partial x}=p-y
\]

这是框架经常融合 Softmax 与 Cross Entropy 的原因之一：既减少中间张量，也能使用更稳定的数学表达。

## 8. 本项目中的三种 Softmax

- Fused Softmax：整行由一个 program 处理，理想情况下输入只读一次。

- Online Softmax：固定大小分块，在线合并 max/sum，输入读取两次。

- softmax(X) @ V 在线融合实验：在更新 max/sum 的同时更新输出 accumulator，不物化概率矩阵。

## 9. 核心记忆

- Softmax 的困难不在除法，而在两个全行 reduction 减最大值解决数值稳定性
- Fused Softmax 解决中间张量流量
- Online Softmax 解决长行工作集过大
- FlashAttention 进一步避免物化分数和概率矩阵
