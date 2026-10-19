# Triton Fused Softmax Kernel

对应代码：`../softmax/kernel.py`。

## 1. Kernel 映射

当前实现采用：

```text
一个 Triton program instance负责矩阵的一行
grid = (n_rows,)
```

program `row_idx` 负责：

```text
x[row_idx, :]
```

行起始地址：

```python
row_start_ptr = input_ptr + row_idx * input_row_stride
```

stride 的单位是元素，不是 Byte。

## 2. BLOCK_SIZE 与 padding

Triton blocked tensor 通常要求规整的静态 shape，因此：

```python
BLOCK_SIZE = triton.next_power_of_2(n_cols)
```

例如：

```text
n_cols=1000 -> BLOCK_SIZE=1024
```

列坐标和 mask：

```python
col_offsets = tl.arange(0, BLOCK_SIZE)
mask = col_offsets < n_cols
```

mask 阻止无效 lane 产生越界 load/store。

## 3. 为什么 masked load 填 `-inf`

```python
row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
```

负无穷同时适配两步 reduction：

```text
max(a, -inf) = a
exp(-inf) = 0
```

若填 0，padding 可能污染最大值和分母。

## 4. Safe Softmax 在 Kernel 中的映射

```python
row_max = tl.max(row, axis=0)
numerator = tl.exp(row - row_max)
denominator = tl.sum(numerator, axis=0)
softmax_output = numerator / denominator
```

`tl.max` 和 `tl.sum` 是 reduction；`tl.maximum` 才是逐元素 maximum。

## 5. blocked tensor 不等于显式 Shared Memory

`tl.load` 返回 Triton blocked tensor。其值通常映射到寄存器等片上资源，但这不等于代码显式申请了 CUDA Shared Memory。

当 tile 过大时，可能出现：

- 寄存器压力增大；
- occupancy 下降；
- register spill；
- 编译资源限制。

因此“一整行只读一次”有资源前提。

## 6. 输入布局契约

当前地址公式是：

\[
\text{address}(i,j)=\text{base}+i\cdot s_0+j
\]

所以要求：

```python
x.stride(1) == 1
```

但不要求整个 Tensor contiguous。行间可以存在间隔，只要 `input_row_stride` 正确。

例如 stride 为 `(4096, 1)`、shape 为 `(128, 1024)` 的 Tensor可以处理；`base[:, ::2]` 的最后一维 stride 为 2，当前 Kernel 不能处理。

## 7. Python 封装职责

封装会检查：

- 输入必须是 CUDA Tensor；
- 必须是二维矩阵；
- dtype 为 FP16/BF16/FP32；
- 最后一维连续；
- 空张量直接返回；
- 超大单行拒绝并提示改用 Online Softmax。

使用明确异常比 Python `assert` 更适合算子 API，因为 `assert` 可能在优化模式中被移除。

## 8. 为什么使用 `empty_like`

输出每个有效位置都会被 Kernel 覆盖，因此：

```python
y = torch.empty_like(x)
```

比 `zeros_like` 更合适。后者会先产生一次无意义的全张量清零写入。

## 9. 性能模型

FP32 理想算法流量：

```text
读取 x：4MN Bytes
写入 y：4MN Bytes
总计：8MN Bytes
```

Semantic Effective GB/s：

\[
\frac{8MN}{\text{latency}}
\]

它是按算子输入输出计算的逻辑带宽，不是 profiler 测得的真实 DRAM 流量。

## 10. 当前实测结论

环境：RTX 5060 Laptop、26 SM、Triton 3.6、FP32、M=4096。

两次独立运行中，Triton Fused Softmax 在主要区间稳定约为：

```text
333 ~ 336 Effective GB/s
```

PyTorch `F.softmax` 对不同 N 存在 shape-dependent 行为；`N=2048 -> 2176` 的性能下降在两次实验中复现。其他孤立低点存在较大运行噪声，不能过度归因。

## 11. 当前限制

- 仅支持二维 row-wise Softmax；
- 没有自动展平 batch/head；
- 没有 `num_warps` 调参；
- BF16、NaN/Inf 语义尚未完整测试；
- 超长行应改用分块算法；
- 当前只有 forward。

## 12. 常见错误

- 错误：masked load 不设置适合 reduction 的 other。
  - 结果：padding 污染 max 或 sum。
- 错误：只给 load 加 mask，store 不加。
  - 结果：最后一个 tile 越界写。
- 错误：接受 stride(1) != 1，却仍然直接 +col_offsets。
  - 结果：静默读取错误元素。
- 错误：把 Effective GB/s 直接当物理 DRAM 带宽。
  - 结果：性能结论失真。
