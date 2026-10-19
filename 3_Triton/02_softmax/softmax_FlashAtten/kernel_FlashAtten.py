import torch
import triton
import triton.language as tl

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
MAX_BLOCK_SIZE = 65536

@triton.jit
def flash_attn_1d_kernel(
    O_ptr, X_ptr, V_ptr,
    stride_xn, 
    stride_vk, stride_vd, 
    stride_on,
    n_cols,
    d_v,
    BLOCK_SIZE: tl.constexpr,
    D_V: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    # 获取 X 矩阵当前行的起始物理指针
    X_row_start_ptr = X_ptr + row_idx * stride_xn
    
    # 初始化三个寄存器级全局变量：Max、Sum、以及我们的输出累加器 Acc！
    m_i = -float('inf')
    l_i = 0.0
    # Acc 是一个一维向量，长度为 V 的特征维度 D_V
    acc = tl.zeros([D_V], dtype=tl.float32)
    
    # 预先生成 V 的特征维度偏移 [0, 1, ..., D_V-1]
    vd_offs = tl.arange(0, D_V)
    # D_V 是向上补齐后的编译期常量；d_v 才是张量真实的列数。
    d_mask = vd_offs < d_v
    
    for block_offset in range(0, n_cols, BLOCK_SIZE):
        # 1. 计算当前的列区间偏移
        cols = block_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        
        # 2. 载入 X 矩阵的一个 Chunk (1D 向量)
        x_ptrs = X_row_start_ptr + cols
        x_chunk = tl.load(x_ptrs, mask=mask, other=-float('inf'))
        
        # 3. 寻找局部的最大值，并更新出新的全局最大值 m_new
        m_ij = tl.max(x_chunk, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        
        # 4. 计算缩放比例 alpha = e^(m_i - m_new)
        alpha = tl.exp(m_i - m_new)
        
        # 5. 计算当前 Block 统一到新基准下的指数项 e^(x_chunk - m_new)
        p_chunk = tl.exp(x_chunk - m_new)
        
        # 6. 更新分母：l_new = l_1 * alpha + l_2_new
        l_i = l_i * alpha + tl.sum(p_chunk, axis=0)
        
        # 7. 读取对应的 V tile，在当前 program 内直接参与加权累加。
        # 注意 V 是二维矩阵，指针需要广播
        # cols[:, None] 把一维的列坐标转成 [BLOCK_SIZE, 1] 的列向量；vd_offs[None, :] 把特征维度偏移转成 [1, D_V] 的行向量
        # 通过 Triton 的广播机制，这两者相加，会自动扩展成一个 [BLOCK_SIZE, D_V] 的二维指针矩阵，指向显存中对应的每一个元素。
        v_ptrs = V_ptr + cols[:, None] * stride_vk + vd_offs[None, :] * stride_vd
        v_mask = mask[:, None] & d_mask[None, :]
        # v_chunk 的形状是 [BLOCK_SIZE, D_V]
        v_chunk = tl.load(v_ptrs, mask=v_mask, other=0.0)
        
        # 8. FlashAttention 核心计算：
        # 把一维的 p_chunk 变成 [BLOCK_SIZE, 1]，去和 [BLOCK_SIZE, D_V] 的 v_chunk 逐行相乘
        scaled_v = p_chunk[:, None] * v_chunk # 利用广播，让概率一维向量 P 乘以 V 矩阵的每一列
        # 然后沿着列方向求和，得到当前 chunk 对结果的贡献 [D_V]
        chunk_out = tl.sum(scaled_v, axis=0)
        
        # 9. 用一模一样的 alpha，去把旧的累加器也缩水！
        acc = acc * alpha + chunk_out
        
        # 更新，继续下一轮
        m_i = m_new

    # 循环结束，一次性拿到结果！直接除以 l_i 完成归一化。
    out = acc / l_i
    
    # 计算写回 O 矩阵的物理地址
    o_ptrs = O_ptr + row_idx * stride_on + vd_offs
    tl.store(o_ptrs, out, mask=d_mask)

def flash_attn_1d(x: torch.Tensor, v: torch.Tensor, BLOCK_SIZE: int = 1024):
    if not isinstance(x, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise TypeError("x 和 v 必须是 torch.Tensor")
    if not x.is_cuda or not v.is_cuda:
        raise ValueError("x 和 v 必须位于 CUDA 设备")
    if x.device != v.device:
        raise ValueError("x 和 v 必须位于同一 CUDA 设备")
    if x.ndim != 2 or v.ndim != 2:
        raise ValueError("x 和 v 必须都是二维矩阵")
    if x.shape[1] != v.shape[0]:
        raise ValueError("v.shape[0] 必须等于 x.shape[1]")
    if x.dtype != v.dtype:
        raise TypeError("x 和 v 的 dtype 必须一致")
    if x.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"不支持 dtype={x.dtype}，仅支持 FP16、BF16 和 FP32")
    if x.stride(1) != 1 or v.stride(1) != 1:
        raise ValueError("x 和 v 的最后一维必须连续")
    if type(BLOCK_SIZE) is not int or BLOCK_SIZE <= 0:
        raise ValueError("BLOCK_SIZE 必须是正整数")
    if BLOCK_SIZE > MAX_BLOCK_SIZE or BLOCK_SIZE & (BLOCK_SIZE - 1):
        raise ValueError("BLOCK_SIZE 必须是不超过 65536 的 2 的幂")

    n_rows, n_cols = x.shape
    d_v = v.shape[1]

    if n_cols == 0:
        raise ValueError("x 的归约维度不能为空")
    
    # 分配输出内存
    out = torch.empty((n_rows, d_v), device=x.device, dtype=x.dtype)

    if n_rows == 0 or d_v == 0:
        return out

    # tl.arange 的长度必须是 2 的幂；真实宽度通过 d_mask 保护。
    D_V_PAD = triton.next_power_of_2(d_v)
    
    grid = (n_rows, )
    
    flash_attn_1d_kernel[grid](
        out, x, v,
        x.stride(0),
        v.stride(0), v.stride(1),
        out.stride(0),
        n_cols, d_v,
        BLOCK_SIZE=BLOCK_SIZE,
        D_V=D_V_PAD,
    )
    return out
