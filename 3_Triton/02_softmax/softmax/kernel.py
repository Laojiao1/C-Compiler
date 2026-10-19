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
