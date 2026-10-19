import torch
import triton
import triton.language as tl

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
MAX_BLOCK_SIZE = 65536

@triton.jit
def softmax_online_kernel(
    output_ptr, input_ptr, 
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # 1. Online 寻找 Max 和全局总和 Sum
    m_i = -float('inf')  # 初始全局 Max
    l_i = 0.0            # 初始全局总和
    
    for block_offset in range(0, n_cols, BLOCK_SIZE):
        # 拿到当前 Chunk 的偏移量
        cols = block_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        
        # 拉取当前 Chunk 数据
        chunk = tl.load(row_start_ptr + cols, mask=mask, other=-float('inf'))
        
        # 1. 找局部 max
        m_ij = tl.max(chunk, axis=0)
        
        # 2. 选出新 max
        m_new = tl.maximum(m_i, m_ij)
        
        # 3. 核心数学公式 (Rescale)
        alpha = tl.exp(m_i - m_new) # 修正值
        l_i = (
            l_i * alpha # 历史贡献
            + tl.sum(tl.exp(chunk - m_new), axis=0) # 当前 Chunk 贡献
        )
        
        # 4. 更新传给下一个 Chunk
        m_i = m_new

    # 此时已经得到全局 m_i 和 l_i；它们是跨 chunk 传递的标量状态。
    
    # 2. 再次分块遍历，利用真正的全局 Max/Sum 算出概率并写回
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    for block_offset in range(0, n_cols, BLOCK_SIZE):
        # 重新生成列坐标与 mask
        cols = block_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        
        # 第二次拉取 Chunk 数据 (第二趟 Global Memory 读取)
        chunk = tl.load(row_start_ptr + cols, mask=mask, other=-float('inf'))
        
        # 计算真正的概率（减去真正的全局大哥 m_i，除以真正的全局总和 l_i）
        probs = tl.exp(chunk - m_i) / l_i
        
        # 将概率分块写回 Global Memory
        tl.store(output_row_start_ptr + cols, probs, mask=mask)

def softmax_online(x: torch.Tensor, BLOCK_SIZE: int = 1024):
    """
    Triton Online Softmax 的 Python 封装
    用固定 BLOCK_SIZE 分块处理长行，避免单个 program 创建超大的 tile
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
    if type(BLOCK_SIZE) is not int or BLOCK_SIZE <= 0:
        raise ValueError("BLOCK_SIZE 必须是正整数")
    if BLOCK_SIZE > MAX_BLOCK_SIZE or BLOCK_SIZE & (BLOCK_SIZE - 1):
        raise ValueError("BLOCK_SIZE 必须是不超过 65536 的 2 的幂")

    n_rows, n_cols = x.shape
    y = torch.empty_like(x)

    if x.numel() == 0:
        return y

    grid = (n_rows, )
    
    softmax_online_kernel[grid](
        y, x,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
