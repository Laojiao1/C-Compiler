import torch
import triton
import triton.language as tl


SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
MAX_FUSED_SIZE = 65536

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    x_row_stride,
    y_row_stride,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # 一个 program 负责一整行，行与行之间没有 reduction 依赖。
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x_row_ptr = x_ptr + row_idx * x_row_stride
    x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)

    # FP16/BF16 的表示范围和有效位不足以稳定承担 reduction，先转成 FP32。
    x = x.to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols

    # padding 位置虽然加载为 0，但减去 mean 后会变成 -mean；必须再次清零，
    # 否则无效位置的 mean^2 会进入方差 reduction。
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    rstd = tl.rsqrt(variance + eps)

    # 将归一化、缩放和平移融合在同一个 kernel 中，不生成全局中间张量。
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = centered * rstd * weight + bias

    y_row_ptr = y_ptr + row_idx * y_row_stride
    tl.store(y_row_ptr + offsets, y, mask=mask)


AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=1),
    triton.Config({}, num_warps=2),
    triton.Config({}, num_warps=4),
    triton.Config({}, num_warps=8),
]

# M/N 都进入 key：特征维度决定单行 reduction，行数会改变全局并行度。
autotuned_layer_norm_kernel = triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["n_rows", "n_cols"],
)(layer_norm_kernel)

# 条件判断
def _validate_layer_norm_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[int, int]:
    """统一 manual、heuristic 和 autotuned launcher 的输入契约。"""
    if not isinstance(x, torch.Tensor):
        raise TypeError("x 必须是 torch.Tensor")
    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight 必须是 torch.Tensor")
    if not isinstance(bias, torch.Tensor):
        raise TypeError("bias 必须是 torch.Tensor")
    if not x.is_cuda or not weight.is_cuda or not bias.is_cuda:
        raise ValueError("x、weight 和 bias 必须位于 CUDA 设备")
    if x.ndim != 2:
        raise ValueError("x 必须是二维矩阵，LayerNorm 沿最后一维计算")
    if weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("weight 和 bias 必须是一维张量")
    if x.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"不支持 dtype={x.dtype}，仅支持 FP16、BF16 和 FP32")
    if weight.dtype != x.dtype or bias.dtype != x.dtype:
        raise TypeError("x、weight 和 bias 的 dtype 必须一致")
    if weight.device != x.device or bias.device != x.device:
        raise ValueError("x、weight 和 bias 必须位于同一 CUDA 设备")
    if x.stride(1) != 1:
        raise ValueError("x 的最后一维必须连续，因为 kernel 使用 offsets 直接寻址")
    if not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("weight 和 bias 必须连续")
    if not isinstance(eps, (int, float)) or eps < 0:
        raise ValueError("eps 必须是非负实数")

    n_rows, n_cols = x.shape
    if weight.numel() != n_cols or bias.numel() != n_cols:
        raise ValueError("weight 和 bias 的长度必须等于 x 的最后一维")
    return n_rows, n_cols

# 根据特征维度选择合适的 block size
def _block_size_for(n_cols: int) -> int:
    block_size = triton.next_power_of_2(n_cols) if n_cols > 0 else 1
    if block_size > MAX_FUSED_SIZE:
        raise ValueError(
            f"特征维度 {n_cols} 超过单个 program 的支持范围；"
            "后续需要改用分块 reduction"
        )
    return block_size


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    num_warps: int = 4,
) -> torch.Tensor:
    # 对二维 CUDA Tensor 的最后一维执行 fused LayerNorm forward。
    n_rows, n_cols = _validate_layer_norm_inputs(x, weight, bias, eps)
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps 必须是 1、2、4 或 8")

    y = torch.empty_like(x)
    if x.numel() == 0:
        return y

    block_size = _block_size_for(n_cols)

    # 当前先固定一行一个 program；后续通过受控实验研究 num_warps 和特征维度。
    grid = (n_rows,)
    layer_norm_kernel[grid](
        x,
        weight,
        bias,
        y,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return y

# 根据 reduction 逻辑宽度选择 program 内的协作 warp 数
def select_num_warps(block_size: int) -> int:
    if block_size <= 0:
        raise ValueError("block_size 必须为正整数")
    return min(max(block_size // 256, 1), 8)

# 根据启发式选择合适的 block size 和 num_warps
def layer_norm_heuristic(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    # 保留已验证并非全局可靠的 BLOCK_SIZE 单调启发式，供复现实验
    _, n_cols = _validate_layer_norm_inputs(x, weight, bias, eps)
    block_size = _block_size_for(n_cols)
    num_warps = select_num_warps(block_size)

    return layer_norm(
        x,
        weight,
        bias,
        eps=eps,
        num_warps=num_warps,
    )

# 自动调优的 LayerNorm
def layer_norm_autotuned(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    # 在 w1/w2/w4/w8 中按 M/N 实测选择配置
    n_rows, n_cols = _validate_layer_norm_inputs(x, weight, bias, eps)

    y = torch.empty_like(x)
    if x.numel() == 0:
        return y

    block_size = _block_size_for(n_cols)
    grid = (n_rows,)
    autotuned_layer_norm_kernel[grid](
        x,
        weight,
        bias,
        y,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
    )
    return y
