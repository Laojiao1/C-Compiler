# Softmax Correctness 与 Benchmark 复盘

## 1. 测试和 Benchmark 的职责不同

- Correctness：用尽量小的数据主动击穿逻辑错误。
- Benchmark：用足够大的数据观察硬件瓶颈和调度行为。

不要用 benchmark 输出代替正确性测试，也不要为了填满 GPU 把单元测试做得巨大。

## 2. Fused Softmax 测试覆盖

当前覆盖：

- `(1,1)` 最小 shape；
- 非 2 的幂列数；
- 对齐列数；
- 尾部 mask；
- FP16/FP32；
- 极大值下的 Safe Softmax；
- 空矩阵；
- 非连续最后一维拒绝。

测试使用 `torch.testing.assert_close`，因为并行 reduction 顺序和 exp 实现可能带来浮点误差。

## 3. Online 测试为什么使用 `4096+123`

```text
n_cols = 4219
BLOCK_SIZE = 256
chunks = 17
```

前 16 个完整 chunk 覆盖 4096 个元素，最后一个 chunk 只有 123 个有效 lane。

一个 shape 同时验证：

- 多次在线 m/l 更新；
- 最后一个 chunk 的 load mask；
- 最后一个 chunk 的 store mask。

如果只测试 `n_cols <= BLOCK_SIZE`，循环只执行一次，alpha rescale 几乎没有得到验证。

## 4. 为什么把 8 放在最后一个 chunk

```python
spike_col = n_cols - 17
x[0, spike_col] = 8.0
```

较大值放在后段，迫使已经积累的历史 `l_i` 使用 alpha 重标定。

不使用 500，是因为：

```text
alpha = exp(old_max - 500)
```

会下溢为 0，历史贡献在数学上也几乎消失。错误实现直接丢弃历史值，可能仍然通过测试。

使用 8 时 alpha 明显非零，历史贡献仍可观察，更容易暴露 rescale Bug。

## 5. 为什么必须断言而不是打印

错误写法：

```python
if torch.allclose(...):
    print("pass")
else:
    print("fail")
```

函数正常返回时，pytest 仍可能报告 PASSED。

正确做法：

```python
torch.testing.assert_close(actual, expected)
```

失败必须通过异常传播给测试框架。

## 6. 概率和检查的作用

\[
\sum_j p_j=1
\]

行和检查有助于定位分母累计和 mask 问题，但不能单独证明 Softmax 正确。均匀错误分布也可能和为 1，所以 reference 逐元素比较仍是主要证据。

## 7. 当前仍可补充的测试

- BF16；
- `n_cols < BLOCK_SIZE`；
- `n_cols == BLOCK_SIZE+1`；
- 多个 BLOCK_SIZE；
- 多个随机 seed；
- NaN/+Inf/-Inf；
- 更长行的累计精度；
- 跨 Triton 版本验证。

## 8. Benchmark 统计口径

使用：

```text
P20：较快分位
P50：中位数，主结果
P80：较慢分位
```

不要把 P20/P80 命名为绝对 min/max。

当前测量包含输出 Tensor 分配，是算子级 latency，不是 profiler 的纯 Kernel time。

## 9. Semantic 与 Algorithmic GB/s

### Semantic Effective GB/s

所有 Softmax 统一按一份输入、一份输出：

\[
\frac{2MNs}{t}
\]

适合比较完成相同语义的速度。

### Online Estimated Algorithmic GB/s

Online 按两次读取、一次写入：

\[
\frac{3MNs}{t}
\]

它是流量模型，不是硬件计数器结果。第二次读取可能命中 L1/L2，因此估算值可以高于 Vector Add 的稳定有效带宽。

## 10. Fused、PyTorch 与 Naive 实测

环境：

```text
RTX 5060 Laptop GPU
26 SM
7.96 GiB VRAM
Compute Capability 12.0
PyTorch 2.11.0+cu128
Triton 3.6.0
FP32
```

`N>=1024` 两次运行平均：

| 实现 | Run 1 | Run 2 |
| --- | ---: | ---: |
| Triton Fused | 330.47 GB/s | 333.07 GB/s |
| PyTorch F.softmax | 276.43 GB/s | 295.95 GB/s |
| PyTorch Naive | 82.91 GB/s | 87.79 GB/s |

Triton 平均绝对变化约 1.82%，表现稳定。PyTorch 和 Naive 波动更大。

## 11. 为什么 Naive 约慢四倍

忽略行级标量，Naive eager 流量粗略为：

```text
max：       4MN
subtract：  8MN
exp：       8MN
sum：       4MN
divide：    8MN
总计：     32MN Bytes（FP32）
```

Fused 理想流量为 8MN，比例约 4。实测 `330/83` 也约等于 4，说明主要收益来自避免中间张量反复读写，而不是 exp 算得更快。

## 12. PyTorch 的 shape-dependent 行为

`N=2048 -> 2176` 的下降在两次运行中复现：

```text
Run 1: 324.0 -> 194.7 GB/s
Run 2: 324.8 -> 197.2 GB/s
```

这说明很可能存在 kernel/configuration 阈值。但具体 dispatch 路径仍需 profiler 或源码证据，不能只靠曲线下结论。

## 13. Online 固定 M 实验

固定 `M=256`、`BLOCK_SIZE=1024`：

| N | Chunks | Online P50 | Speedup vs PyTorch |
| ---: | ---: | ---: | ---: |
| 4096 | 4 | 0.0345 ms | 1.241x |
| 8192 | 8 | 0.0698 ms | 0.887x |
| 16384 | 16 | 0.1546 ms | 0.755x |
| 32768 | 32 | 0.3127 ms | 1.246x |
| 65536 | 64 | 0.6589 ms | 0.747x |
| 131072 | 128 | 1.3398 ms | 0.789x |

多数长行上 Online 慢约 20%~25%。个别反超点可能来自 PyTorch shape 路径或运行波动，不能推广。

## 14. Online 固定总元素实验

固定 `M*N=2^24`：

| Shape | Chunks/row | Online P50 | Semantic GB/s |
| --- | ---: | ---: | ---: |
| `4096x4096` | 4 | 0.4446 ms | 301.88 |
| `256x65536` | 64 | 0.6140 ms | 218.61 |
| `64x262144` | 256 | 0.7256 ms | 184.97 |
| `16x1048576` | 1024 | 1.4280 ms | 93.99 |

首尾总元素相同，但少行长列约慢 3.21 倍。原因是 program 数从 4096 降到 16，同时单行串行 chunk 从 4 增加到 1024。

## 15. 怎样写严谨的性能结论

应该说：在指定 GPU、dtype、shape 和软件版本下，两次运行观察到某趋势。

不应该说：

- Triton 永远比 PyTorch 快。
- 某个 Effective GB/s 就是物理 DRAM 峰值。
- 一个孤立低点一定是框架切换了某个 Kernel。

性能优化必须同时给出环境、延迟、分位数、流量口径、正确性和重复性。
