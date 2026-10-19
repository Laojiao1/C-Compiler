import math

import torch
import triton
import triton.language as tl


SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
SUPPORTED_BLOCK_SIZES = (16, 32, 64)
SUPPORTED_NUM_WARPS = (2, 4, 8)


@triton.jit
def _flash_attention_forward_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    sequence_length,
    head_dim,
    num_heads,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    scale,
    CAUSAL: tl.constexpr,
    SKIP_CAUSAL_FUTURE_TILES: tl.constexpr,
    INPUT_TYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """一个 Program 固定一个 batch/head/query tile，并遍历所有 K/V tile。"""
    # Grid：(query tile, batch*head), Key tile 是 Program 内部循环
    query_block_idx = tl.program_id(0)
    batch_head_idx = tl.program_id(1) # B*N

    # 把 B*N 映射回来
    batch_idx = batch_head_idx // num_heads
    head_idx = batch_head_idx % num_heads

    query_offsets = query_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    feature_offsets = tl.arange(0, BLOCK_D)

    query_mask = query_offsets < sequence_length
    feature_mask = feature_offsets < head_dim

    q_base = q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = k_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = v_ptr + batch_idx * stride_vb + head_idx * stride_vh

    # 加载数据 Q tile [BLOCK_M, BLOCK_D] 在所有 K/V 循环中复用，只加载一次
    q_ptrs = q_base + query_offsets[:, None] * stride_qn + feature_offsets[None, :] * stride_qd
    q_tile = tl.load(q_ptrs,
        mask=query_mask[:, None] & feature_mask[None, :],
        other=0.0
    )

    # 每个 query 独立维护在线 Softmax 状态：
    # m = max(score)，l = sum(exp(score-m))，acc = sum(exp(score - m) * V)。
    m_running_max = tl.where(query_mask, -float("inf"), 0.0).to(tl.float32)
    denominator = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # 处理因果跳块
    key_block_count = tl.cdiv(sequence_length, BLOCK_N)
    if CAUSAL:
        if SKIP_CAUSAL_FUTURE_TILES:
            # 只遍历起点不超过当前 Query Tile 最大位置的 Key Tile
            query_tile_end = tl.minimum(
                (query_block_idx + 1) * BLOCK_M,
                sequence_length,
            )
            key_block_count = tl.cdiv(query_tile_end, BLOCK_N)

    key_offsets_in_block = tl.arange(0, BLOCK_N)
    for key_block_idx in range(0, key_block_count):
        key_offsets = key_block_idx * BLOCK_N + key_offsets_in_block
        key_mask = key_offsets < sequence_length

        # 通过索引广播直接加载 K^T tile [BLOCK_D,BLOCK_N]
        k_ptrs = (
            k_base
            + feature_offsets[:, None] * stride_kd
            + key_offsets[None, :] * stride_kn
        )
        v_ptrs = (
            v_base
            + key_offsets[:, None] * stride_vn
            + feature_offsets[None, :] * stride_vd
        )

        # 加载数据 K^T tile 和 v tile
        k_T_tile = tl.load(
            k_ptrs,
            mask=feature_mask[:, None] & key_mask[None, :],
            other=0.0,
        )
        v_tile = tl.load(
            v_ptrs,
            mask=key_mask[:, None] & feature_mask[None, :],
            other=0.0,
        )

        # 计算 Score 在当前循环直接计算出结果，不写回显存
        scores = tl.dot(q_tile, k_T_tile, input_precision="ieee") * scale
        score_mask = query_mask[:, None] & key_mask[None, :]

        # 处理因果跳块
        if CAUSAL:
            # 行 i query，列 j key, 只保留 j <= i
            score_mask = score_mask & (key_offsets[None, :] <= query_offsets[:, None])
        scores = tl.where(score_mask, scores, -float("inf"))

        # 维护最大值 m_max
        m_block_max = tl.max(scores, axis=1)
        m_new_max = tl.maximum(m_running_max, m_block_max)
        # 最大值变化后，旧 l/acc 必须乘同一个 alpha，切换到新基准
        alpha = tl.exp(m_running_max - m_new_max)

        # 计算当前块的未归一化权重
        numerator = tl.exp(scores - m_new_max[:, None])
        numerator = tl.where(score_mask, numerator, 0.0)

        # l_new = l_old * alpha + sum(numerator)
        denominator = denominator * alpha + tl.sum(numerator, axis=1)
        acc = acc * alpha[:, None]

        # 权重转为输入 dtype 进入 tl.dot，输出仍累加到 FP32 acc
        if INPUT_TYPE == 0:
            acc = tl.dot(
                numerator.to(tl.float16),
                v_tile,
                acc,
            )
        elif INPUT_TYPE == 1:
            acc = tl.dot(
                numerator.to(tl.bfloat16),
                v_tile,
                acc,
            )
        else:
            acc = tl.dot(
                numerator,
                v_tile,
                acc,
                input_precision="ieee",
            )
        m_running_max = m_new_max

    # acc 是未归一化加权和，除以 l 后得到 output = softmax(score)@V
    safe_denominator = tl.where(denominator > 0.0, denominator, 1.0)
    output_tile = acc / safe_denominator[:, None]

    output_ptrs = (
        output_ptr
        + batch_idx * stride_ob
        + head_idx * stride_oh
        + query_offsets[:, None] * stride_on
        + feature_offsets[None, :] * stride_od
    )
    tl.store(
        output_ptrs,
        output_tile,
        mask=query_mask[:, None] & feature_mask[None, :]
    )


"""输入检查"""
def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[int, int, int, int]:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} 必须是 torch.Tensor")
        if not tensor.is_cuda:
            raise ValueError(f"{name} 必须位于 CUDA 设备")
        if tensor.ndim != 4:
            raise ValueError(f"{name} 必须是四维张量 [B, H, N, D]")

    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q、k、v 的形状必须完全一致")
    if q.dtype not in SUPPORTED_DTYPES:
        raise TypeError("仅支持 FP16、BF16 和 FP32")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q、k、v 的 dtype 必须一致")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q、k、v 必须位于同一 CUDA 设备")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("q、k、v 的最后一维必须连续")

    batch_size, num_heads, sequence_length, head_dim = q.shape
    if head_dim == 0 or head_dim > 128:
        raise ValueError("当前版本要求 head_dim 位于 [1, 128] 范围内")

    return batch_size, num_heads, sequence_length, head_dim


"""根据当前实验结论决定正式入口是否启用因果跳块"""
def _should_skip_causal_future_tiles(
    sequence_length: int,
    causal: bool,
) -> bool:
    return causal and sequence_length >= 512


"""使用当前实验最优统一配置的 FlashAttention Forward"""
def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    
    batch_size, num_heads, sequence_length, head_dim = _validate_inputs(q, k, v)
    if not isinstance(causal, bool):
        raise TypeError("causal 必须是 bool")

    # 当前硬件实验中，N>=512 才观察到稳定跳块收益
    skip_causal_future_tiles = _should_skip_causal_future_tiles(
        sequence_length,
        causal,
    )
    return _launch_flash_attention(
        q,
        k,
        v,
        batch_size,
        num_heads,
        sequence_length,
        head_dim,
        causal=causal,
        block_m=64,
        block_n=32,
        num_warps=4,
        skip_causal_future_tiles=skip_causal_future_tiles,
    )


"""显式指定 Tile/warp 的实验入口，供控制变量调优使用"""
def flash_attention_forward_configured(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    block_m: int = 32,
    block_n: int = 32,
    num_warps: int = 4,
    skip_causal_future_tiles: bool = False,
) -> torch.Tensor:
    batch_size, num_heads, sequence_length, head_dim = _validate_inputs(q, k, v)
    if not isinstance(causal, bool):
        raise TypeError("causal 必须是 bool")
    if not isinstance(skip_causal_future_tiles, bool):
        raise TypeError("skip_causal_future_tiles 必须是 bool")
    if skip_causal_future_tiles and not causal:
        raise ValueError("只有 causal=True 时才能跳过完全未来的 Key Tile")
    if block_m not in SUPPORTED_BLOCK_SIZES:
        raise ValueError("block_m 必须是 16、32 或 64")
    if block_n not in SUPPORTED_BLOCK_SIZES:
        raise ValueError("block_n 必须是 16、32 或 64")
    if num_warps not in SUPPORTED_NUM_WARPS:
        raise ValueError("num_warps 必须是 2、4 或 8")

    return _launch_flash_attention(
        q,
        k,
        v,
        batch_size,
        num_heads,
        sequence_length,
        head_dim,
        causal=causal,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        skip_causal_future_tiles=skip_causal_future_tiles,
    )


"""使用已验证的输入元数据启动 Kernel，避免不同入口重复验证"""
def _launch_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    num_heads: int,
    sequence_length: int,
    head_dim: int,
    causal: bool,
    block_m: int,
    block_n: int,
    num_warps: int,
    skip_causal_future_tiles: bool,
) -> torch.Tensor:
    
    output = torch.empty_like(q)
    if sequence_length == 0:
        return output

    # D 补齐到不小于 16 的 2 次幂，padding 特征由 mask 填 0。
    block_d = max(16, triton.next_power_of_2(head_dim))
    # Key tile 不进入 Grid；同一 Program 遍历全部 Key tile 并维护 m/l/acc。

    grid = (
        triton.cdiv(sequence_length, block_m),
        batch_size * num_heads,
    )
    input_type = {
        torch.float16: 0,
        torch.bfloat16: 1,
        torch.float32: 2,
    }[q.dtype]

    _flash_attention_forward_kernel[grid](
        q,
        k,
        v,
        output,
        sequence_length,
        head_dim,
        num_heads,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        1.0 / math.sqrt(head_dim),
        CAUSAL=causal,
        SKIP_CAUSAL_FUTURE_TILES=skip_causal_future_tiles,
        INPUT_TYPE=input_type,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return output
