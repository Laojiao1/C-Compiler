# 02_softmax 复盘

## 1. 当前包含的三个实验

| 实现 | 计算内容 | 输入读取 | 主要目的 |
| --- | --- | --- | --- |
| `kernel.py` | Row-wise Fused Softmax | 一次 | 学习 reduction、数值稳定性和整行 blocked tensor |
| `kernel_online.py` | 2-Pass Online Softmax | 两次 | 用固定 tile 处理长行，学习在线更新 `m_i/l_i` |
| `kernel_FlashAtten.py` | `softmax(X) @ V` | `X` 一次，并按输出行读取 `V` | 学习在线更新输出 accumulator，不物化概率矩阵 |

第三个实验不是完整 FlashAttention：它接收已经物化的 `X = QK^T`，没有在 tile 内计算 `QK^T`，也没有 batch/head、scale、causal mask 或 backward。

## 2. Fused Softmax

对一行向量 `x`，数值稳定的 Softmax 为：

```text
m = max(x)
p = exp(x - m)
y = p / sum(p)
```

减去最大值后，最大的指数输入为 0，避免正向指数溢出。Kernel 将一整行映射成长度为 `next_power_of_2(n_cols)` 的 blocked tensor，padding lane 用 `-inf` 填充：

- `max(-inf)` 不会改变有效最大值；
- `exp(-inf) = 0`，不会改变分母；
- store mask 阻止 padding lane 写回。

中间的 `max/exp/sum/div` 没有显式写回 Global Memory。blocked tensor 通常驻留在寄存器；行过长时寄存器压力会增大，并可能 spill，因此当前封装对单 program tile 设置了上限。

### 算法字节量

对 FP32 的理想 fused 实现：

```text
读取 x：4 * M * N Bytes
写入 y：4 * M * N Bytes
最小算法字节量：8 * M * N Bytes
```

由此得到逻辑有效带宽：

```text
effective_GB/s = 8 * M * N / latency_ms * 1e-6
```

这不是对真实 DRAM 流量的直接测量。PyTorch CUDA Softmax 本身也可能使用融合和特化 kernel，不能假设它一定执行三个独立 kernel 或产生 24 Bytes/element 的 DRAM 流量。

## 3. Online Softmax

当一行不能作为单个大 tile 处理时，把它切成多个 chunk。假设旧状态为 `(m_i, l_i)`，当前 chunk 的最大值为 `m_ij`：

```text
m_new = max(m_i, m_ij)
alpha = exp(m_i - m_new)
l_new = l_i * alpha + sum(exp(chunk - m_new))
```

修正项来自：

```text
exp(x - m_new) = exp(x - m_i) * exp(m_i - m_new)
```

因此旧 chunk 的原始元素不需要为了更新全局最大值而重新读取。

但是，普通 Softmax 最终必须输出每一个概率。第一趟结束后只保存了标量 `m_i/l_i`，早期 chunk 的逐元素值已经丢失，所以仍需第二次读取输入：

```text
第一趟：读取 x，计算全局 m_i/l_i
第二趟：再次读取 x，计算并写出概率
```

对 FP32，其最小算法字节量是：

```text
两次读取 x：8 * M * N Bytes
一次写入 y：4 * M * N Bytes
总计：12 * M * N Bytes
```

所以 Online Softmax 解决的是单 tile 容量和数值在线更新问题，不是把物化 Softmax 降成一次输入读取。

## 4. 在线融合 `softmax(X) @ V`

如果最终目标不是输出概率矩阵，而是：

```text
O = softmax(X) @ V
```

就可以在处理每个 `X` chunk 时直接读取对应 `V` tile，并维护未归一化 accumulator：

```text
acc_new = acc_old * alpha + sum(exp(x_chunk - m_new)[:, None] * V_chunk)
l_new   = l_old   * alpha + sum(exp(x_chunk - m_new))
O       = acc / l
```

当全局最大值改变时，旧分母和旧输出 accumulator 必须使用同一个 `alpha` 重标定。这使得中间概率矩阵无需写入 Global Memory。

当前实现为了满足 `tl.arange`，把 `d_v` 向上补齐到 2 的幂。真实宽度由 `d_mask` 同时保护：

- `V` 的 padding 列不能被读取；
- `O` 的 padding 列不能被写入。

需要注意：当前每个输出行都会遍历一次完整的 `V`。它展示了数学融合，但还没有真正 FlashAttention 对 `Q/K/V` 的二维 tiling 和跨 query tile 数据复用。

## 5. Correctness 闭环

2026-07-17 在当前 CUDA/Triton 环境中手动执行：

- Fused Softmax：`10 passed`；
- Online Softmax：`2 passed`；
- 在线融合实验：`4 passed`。

覆盖内容包括：

- FP16、FP32 Fused Softmax；
- 最小、对齐和非 2 次幂列数；
- 空矩阵；
- 非连续最后一维拒绝；
- Online Softmax 的非对齐 chunk 和非零 `alpha` 重标定；
- `d_v=63/64/65`，验证补齐维度不会越界；
- 不兼容 shape 和非法 `BLOCK_SIZE` 拒绝。

仍待补充：BF16、NaN/Inf 语义、更多随机 shape，以及跨 Triton 版本测试。

## 6. Benchmark 口径

下一轮 Softmax benchmark 应至少对比：

```text
PyTorch F.softmax
Triton Fused Softmax
Triton Online Softmax（长行场景）
```

必须同时报告：

- latency，而不是只报告 GB/s；
- 基于输入输出语义的 effective GB/s；
- Fused 与 Online 不同的算法读取次数；
- shape、dtype、GPU 和软件版本；
- correctness 是否在 benchmark 前独立通过。

直接判断性能优劣时以 latency/speedup 为准。GB/s 用来解释瓶颈，但不能用错误的字节模型制造“性能碾压”。

`kernel_FlashAtten.py` 当前只作为数学教学实验，不与 PyTorch SDPA 做正式 FlashAttention 性能对比。真正的对比留到 `06_flash_attention_forward`。

## 7. Fused Softmax Baseline 实测

### 7.1 测试环境

```text
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
SM: 26
VRAM: 7.96 GiB
Compute Capability: 12.0
PyTorch: 2.11.0+cu128
Triton: 3.6.0
PyTorch CUDA: 12.8
dtype: FP32
warmup/rep: 100/300 ms
M: 4096
N: 128 ~ 8192，步长 128
```

对比对象：

```text
Triton Fused Softmax
PyTorch F.softmax
PyTorch Naive: max -> subtract -> exp -> sum -> divide
```

### 7.2 两次独立运行的重复性

在 `N >= 1024` 的区间：

| 实现 | Run 1 平均 Effective GB/s | Run 2 平均 Effective GB/s |
| --- | ---: | ---: |
| Triton Fused | 330.47 | 333.07 |
| PyTorch F.softmax | 276.43 | 295.95 |
| PyTorch Naive | 82.91 | 87.79 |

所有 shape 的平均绝对变化：

```text
Triton Fused:       1.82%
PyTorch F.softmax:  9.20%
PyTorch Naive:      8.48%
```

Triton 在 Run 1 中出现过 `4480 ~ 4736` 和 `7296 ~ 7936` 两段局部下降，但 Run 2 中这些下降基本消失；Run 2 又在 `N=6272` 出现一个孤立低点。因此这些位置更可能受到动态频率、Laptop 功耗状态、WSL/桌面调度等运行状态影响，而不是固定 shape 对应的必然退化。

保守结论是：当前 Triton Fused Softmax 在主要区间稳定在约 `333 ~ 336 Effective GB/s`。

### 7.3 Fusion 收益与流量模型吻合

忽略只有 `M` 个元素的行级标量，Naive Softmax 的主要显存流量可以粗略拆解为：

```text
max:       读取 x                              ~= 4MN Bytes
subtract:  读取 x + 写入临时张量               ~= 8MN Bytes
exp:       读取临时张量 + 写入 numerator        ~= 8MN Bytes
sum:       读取 numerator                      ~= 4MN Bytes
divide:    读取 numerator + 写入 output         ~= 8MN Bytes
总计                                           ~= 32MN Bytes
```

Fused Softmax 的理想算法流量为：

```text
读取 x + 写入 output ~= 8MN Bytes
```

因此理论流量比例约为：

```text
32MN / 8MN = 4
```

实测中 Triton Fused 约为 `330 ~ 333 GB/s`，Naive 约为 `83 ~ 88 GB/s`，性能比接近 4。这说明 Naive 版本的主要损失来自中间张量反复读写 Global Memory，而不是 `exp` 本身算得慢。

图中三条曲线统一使用 `8MN Bytes` 计算 Effective GB/s，目的是比较完成同一 Softmax 语义所需的时间。Naive 曲线的数值不是其真实 DRAM 吞吐率。

### 7.4 PyTorch 的 shape-dependent 行为

最稳定复现的性能阈值出现在 `N=2048 -> 2176`：

```text
Run 1: 324.0 -> 194.7 GB/s
Run 2: 324.8 -> 197.2 GB/s
```

随后 PyTorch 的有效带宽随 N 增大逐步恢复。这说明 PyTorch 很可能在该边界选择了不同的 kernel 或 launch 配置，新路径在区间起点的 tile 利用率较低。

但 `N=4352/4480/8192` 等位置的极端低点没有在两次运行中稳定复现，所以不能全部归因于内部 dispatch。要确认具体原因，需要 profiler 中的 kernel 名称、launch 配置或 PyTorch CUDA Softmax 源码证据。

### 7.5 当前阶段结论与延期项

已经完成：

- Fused Softmax 的输入契约和 correctness 测试；
- 非 2 次幂 mask、空矩阵、FP16/FP32 和非连续布局测试；
- Online Softmax 的在线 `m_i/l_i` 正确性；
- `softmax(X) @ V` 实验中 `m_i/l_i/acc` 的重标定；
- Triton、PyTorch F.softmax、PyTorch Naive 三方 baseline；
- 两次独立运行的重复性分析；
- Fusion 收益与显存流量模型的相互验证。

复盘后再继续：

- `num_warps=4/8/16` 的受控实验；
- reduction 的跨 warp 代价、寄存器压力与 occupancy；
- Online Softmax 的长行 benchmark；
- 使用 profiler 验证 PyTorch 的 shape-dependent kernel 路径；
- BF16、NaN/Inf 与更多随机 shape。

`01_vector_add` 和 `02_softmax` 至此进入暂停复盘阶段。在完成代码逐行讲解和个人复盘前，不继续进入 `03_matmul`。

## kernel 代码

```py
import torch
import triton
import triton.language as tl

# Triton Fused Softmax 的 Python 封装
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# 限制最大列数
MAX_FUSED_SIZE = 65536

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, 
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 第一步：拿到当前行的行号
    # 一个 Block 承包一整行的工作
    row_idx = tl.program_id(0)
    
    # 第二步：计算当前行在 Global Memory 中的物理起始地址
    # 这就是 stride 的威力：逻辑上的下一行，物理上要跳过 input_row_stride 个元素
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # 第三步：拿到属于这一行的偏移量 [0, 1, 2, ..., BLOCK_SIZE-1]
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # 掩码保护：如果矩阵的列数不是 BLOCK_SIZE 的整数倍，挡住越界部分
    mask = col_offsets < n_cols
    
    # 第四步：把整行映射成一个 blocked tensor，只从 Global Memory 读取一次。
    # 这些值通常驻留在寄存器；tile 过大会造成寄存器压力，甚至发生 spill。
    # other=-float('inf') 是一个非常暴力的技巧：
    # 对于越界的无效位，我们填入负无穷。这样下面找 Max 时它绝对选不上，
    # 算 exp(-inf) 时它直接变 0，完美不影响后续的总和（Sum）计算！
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # 后续 max、exp、sum 不再显式访问 Global Memory，中间值由编译器放在片上资源中。
    
    # 第五步：Safe Softmax 
    # 找出一整行的最大值。axis=0 是因为拉进来的 row 是个一维向量
    row_max = tl.max(row, axis=0)
    
    # 第六步：每个元素减去最大值，并算指数 e
    numerator = tl.exp(row - row_max)
    
    # 第七步：Safe Softmax 第二式 (算总和)
    # 把算出来的 e 全部加起来
    denominator = tl.sum(numerator, axis=0)
    
    # 第八步：算除法得到最终的概率分布
    softmax_output = numerator / denominator
    
    # 第九步：计算输出地址，并将最终结果写回 Global Memory。
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x: torch.Tensor):
    """
    Triton Fused Softmax 的 Python 封装
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x 必须是 torch.Tensor")
    if not x.is_cuda:
        raise ValueError("x 必须位于 CUDA 设备")
    if x.ndim != 2:
        raise ValueError("x 必须是二维矩阵，Softmax 沿最后一维计算")
    if x.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"不支持 dtype={x.dtype}，仅支持 FP16、BF16 和 FP32")
    if x.stride(1) != 1:
        raise ValueError("x 的最后一维必须连续，因为 Kernel 使用 col_offset 直接寻址")

    n_rows, n_cols = x.shape

    # 先创建输出，空矩阵不需要启动 GPU Kernel。
    y = torch.empty_like(x)
    if x.numel() == 0:
        return y
    
    # 坑：Triton 对一块 Block 处理的最大数据量有限制
    # 为了简化，我们假设矩阵的列数可以用下一个 2 的幂次容纳
    # 比如列数是 1000，我们要开一个 BLOCK_SIZE=1024 的坑
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise ValueError(
            f"列数 {n_cols} 超过单个 program 的支持范围；"
            "请改用分块 Online Softmax"
        )
    
    # BLOCK_SIZE 越大，需要同时保持的寄存器值越多；超过上限时改走分块算法。
    
    # 定义网格：矩阵有多少行，就派出多少个 Block！
    # 注意这里不用切分了，因为我们规定一个 Block 吞下一整行
    grid = (n_rows, )
    
    # 启动 Kernel！
    # x.stride(0) 就是极其关键的 input_row_stride
    softmax_kernel[grid](
        y, x,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y

```















