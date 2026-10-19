import math

import torch
import triton
import triton.language as tl


SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
MAX_SOFTMAX_SIZE = 65536


"""
    步骤一：计算 Scores = (Q @ K^T) * scale
    其中 acc = Q @ K^T
"""
@triton.jit
def _attention_scores_kernel(
    q_ptr, k_ptr, scores_ptr, # 指针
    sequence_length, # 序列长度，用于边界检查
    head_dim, # 注意力头维度
    num_heads, # 注意力头数量
    # Q K S 张量的 Strides，QK 形状都为 [B, H, N, D], S 形状为 [B, H, N, N]
    # [Batch, Head, NumberOfToken, Dim] -> [B, H, N, D]
    # 含义：[BatchSize(几句话), 头数量(几个不同的语义视角), 词数量(一句话有几个词), 单头特征维度(每个词用多少个数字描述)]
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_sb, stride_sh, stride_sm, stride_sn,
    scale, # 注意力缩放因子
    CAUSAL: tl.constexpr, # 是否开启因果跳块掩码
    BLOCK_M: tl.constexpr, # 对应 Q 中的维度 N
    BLOCK_N: tl.constexpr, # 对应 K 中的维度 N
    BLOCK_D: tl.constexpr, # 对应 Q K 中的维度 D
):
    """计算一个 batch/head 中的分数矩阵小块。"""
    query_block_idx = tl.program_id(0) # 负责 S 中，行方向（Query 词序列）上的小块
    key_block_idx = tl.program_id(1) # 负责 S 中，列方向（key 词序列）上的小块
    batch_head_idx = tl.program_id(2) # 负责第几句话中的第几个注意力头

    # 第三维 grid 把 batch 和 head 压成一个编号，再在 kernel 中还原
    batch_idx = batch_head_idx // num_heads
    head_idx = batch_head_idx % num_heads

    query_offsets = query_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    key_offsets = key_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    inner_offsets = tl.arange(0, BLOCK_D)

    q_base = q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = k_ptr + batch_idx * stride_kb + head_idx * stride_kh

    # 每个 program 负责 scores 的 [BLOCK_M, BLOCK_N] 小块；
    # head_dim 与 MatMul 的 K 维相同，只增加 program 内部的归约循环
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for inner_block_idx in range(0, tl.cdiv(head_dim, BLOCK_D)):
        current_inner_offsets = inner_block_idx * BLOCK_D + inner_offsets
        # 计算 Q/K 小块的全局指针
        q_ptrs = (
            q_base +
            query_offsets[:, None] * stride_qn +
            current_inner_offsets[None, :] * stride_qd
        )
        # K 的小块需要转置才能与 Q 做矩阵乘法。这里直接在索引上实现转置，而不是在显存中物化转置后的 K
        k_transposed_ptrs = (
            k_base + 
            current_inner_offsets[:, None] * stride_kd +
            key_offsets[None, :] * stride_kn
        )

        # 使用掩码避免越界访问
        q_mask = (
            (query_offsets[:, None] < sequence_length) &
            (current_inner_offsets[None, :] < head_dim)
        )
        k_mask = (
            (current_inner_offsets[:, None] < head_dim) &
            (key_offsets[None, :] < sequence_length)
        )

        # 加载数据
        q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)
        k_transposed_tile = tl.load(k_transposed_ptrs, mask=k_mask, other=0.0)

        # Q @ K^T 乘累加运算，精度选择 ieee fp32，避免后续作比较时的参考误差
        accumulator = tl.dot(q_tile, k_transposed_tile, accumulator, input_precision="ieee")

    # scores = acc * scale
    scores = accumulator * scale

    # 如果启动了 causal，需要进行配置跳块掩码
    if CAUSAL:
        # 只能看到 j <= i 部分
        allowed = key_offsets[None, :] <= query_offsets[:, None]
        # 三元表达式，j > i 的部分初始化为 -inf，在后面做指数运算时就变成了0
        scores = tl.where(allowed, scores, -float("inf"))

    score_ptrs = (
        scores_ptr
        + batch_idx * stride_sb
        + head_idx * stride_sh
        + query_offsets[:, None] * stride_sm
        + key_offsets[None, :] * stride_sn
    )
    score_mask = (
        (query_offsets[:, None] < sequence_length)
        & (key_offsets[None, :] < sequence_length)
    )

    # 写回 scores
    tl.store(score_ptrs, scores, mask=score_mask)


"""
    步骤二：计算 softmax(scores)
    分四步：1. 计算每行最大 m_max
           2. 计算分子 numerator = exp(x - m_max)
           3. 计算分母 denominator = sum(exp(x - m_max))
           4. 计算结果 p = numerator / denominator
"""
@triton.jit
def _attention_softmax_kernel(
    scores_ptr,
    probabilities_ptr,
    sequence_length,
    num_heads,
    # scores 维度为 [BatchSize, Head, Query, Key] -> [B, H, N, N]
    # 含义：[BatchSize(几句话), 头数量(几个不同的语义视角), 第几个 Query 词, 所有被关注的 Key 词]
    stride_sb, stride_sh, stride_sq, stride_sk,
    stride_pb, stride_ph, stride_pq, stride_pk,
    BLOCK_SIZE: tl.constexpr,
):
    """一个 program 负责一个 batch/head/query 对应的完整分数行。"""
    row_idx = tl.program_id(0)

    # 总行数为 B*H*N，这里将线性行号还原成对应的四个维度 [B, H, N, N] 
    query_idx = row_idx % sequence_length
    batch_head_idx = row_idx // sequence_length
    head_idx = batch_head_idx % num_heads
    batch_idx = batch_head_idx // num_heads

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < sequence_length

    score_row_ptr = (
        scores_ptr
        + batch_idx * stride_sb
        + head_idx * stride_sh
        + query_idx * stride_sq
    )

    # 加载分数 scores
    score_row = tl.load(score_row_ptr + offsets * stride_sk, 
                        mask=mask, other=-float("inf")).to(tl.float32)

    # safe-softmax：减去行最大值，避免溢出
    m_max = tl.max(score_row, axis=0)
    numerator = tl.exp(score_row - m_max)
    denominator = tl.sum(numerator, axis=0)
    p = numerator / denominator

    p_row_ptr = (
        probabilities_ptr
        + batch_idx * stride_pb
        + head_idx * stride_ph
        + query_idx * stride_pq
    )

    # 写回 p
    tl.store(p_row_ptr + key_offsets * stride_pk, p, mask=mask)


"""
    步骤三：计算结果 output = p @ v
"""
@triton.jit
def _attention_output_kernel(
    p_ptr, v_ptr, output_ptr,
    sequence_length,
    head_dim,
    num_heads,
    stride_pb, stride_ph, stride_pq, stride_pk, # p [B, H, N, N]
    stride_vb, stride_vh, stride_vn, stride_vd, # v [B, H, N, D]
    stride_ob, stride_oh, stride_on, stride_od, # output [B, H, N, D]
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_block_idx = tl.program_id(0) # 负责 output 中，行方向（Query 词序列）上的小块
    feature_block_idx = tl.program_id(1) # 负责 output 中，列方向（特征维度 D 词序列）上的小块
    batch_head_idx = tl.program_id(2)  # 负责第几句话中的第几个注意力头

    # 同理，映射还原
    batch_idx = batch_head_idx // num_heads
    head_idx = batch_head_idx % num_heads

    query_offsets = query_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    feature_offsets = feature_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    reduction_offsets = tl.arange(0, BLOCK_K)

    p_base = p_ptr + batch_idx * stride_pb + head_idx * stride_ph
    v_base = v_ptr + batch_idx * stride_vb + head_idx * stride_vh

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for reduction_block_idx in range(0, tl.cdiv(sequence_length, BLOCK_K)):
        cur_reduction_offsets = reduction_block_idx * BLOCK_K + reduction_offsets

        p_ptrs = p_base + query_offsets[:, None] * stride_pq + cur_reduction_offsets[None, :] * stride_pk
        v_ptrs = v_base + cur_reduction_offsets[:, None] * stride_vn + feature_offsets[None, :] * stride_vd
        
        p_mask = (
            (query_offsets[:, None] < sequence_length)
            & (cur_reduction_offsets[None, :] < sequence_length)
        )
        v_mask = (
            (cur_reduction_offsets[:, None] < sequence_length)
            & (feature_offsets[None, :] < head_dim)
        )

        # 加载 p v
        p_tile = tl.load(p_ptrs, mask=p_mask, other=0.0)
        v_tile = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # 累加计算 p @ v
        acc = tl.dot(p_tile, v_tile, acc, input_precision="ieee")

    output_ptrs = (
        output_ptr
        + batch_idx * stride_ob
        + head_idx * stride_oh
        + query_offsets[:, None] * stride_on
        + feature_offsets[None, :] * stride_od
    )
    output_mask = (
        (query_offsets[:, None] < sequence_length)
        & (feature_offsets[None, :] < head_dim)
    )

    # 写回结果
    tl.store(output_ptrs, acc, mask=output_mask)


"""
    输入 score 数据格式检查
"""
def _validate_attention_score_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[int, int, int, int]:
    """检查分数 Kernel 当前支持的 Q/K 输入契约。"""
    for name, tensor in (("q", q), ("k", k)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} 必须是 torch.Tensor")
        if not tensor.is_cuda:
            raise ValueError(f"{name} 必须位于 CUDA 设备")
        if tensor.ndim != 4:
            raise ValueError(f"{name} 必须是四维张量 [B, H, N, D]")

    if q.shape != k.shape:
        raise ValueError("q 和 k 的形状必须一致，均为 [B, H, N, D]")
    if q.dtype not in SUPPORTED_DTYPES:
        raise TypeError(
            f"不支持 dtype={q.dtype}，仅支持 FP16、BF16 和 FP32"
        )
    if q.dtype != k.dtype:
        raise TypeError("q 和 k 的 dtype 必须一致")
    if q.device != k.device:
        raise ValueError("q 和 k 必须位于同一 CUDA 设备")

    batch_size, num_heads, sequence_length, head_dim = q.shape
    if head_dim == 0:
        raise ValueError("head_dim 必须大于 0，才能计算缩放因子 1/sqrt(D)")
    return batch_size, num_heads, sequence_length, head_dim


"""
    _attention_scores_kernel 启动函数
"""
def attention_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """使用 Triton 显式计算并写出 S = QK^T / sqrt(D)。"""
    batch_size, num_heads, sequence_length, head_dim = (
        _validate_attention_score_inputs(q, k)
    )
    if not isinstance(causal, bool):
        raise TypeError("causal 必须是 bool")

    scores = torch.empty(
        (batch_size, num_heads, sequence_length, sequence_length),
        device=q.device,
        dtype=q.dtype,
    )
    if scores.numel() == 0:
        return scores

    block_m = 32
    block_n = 32
    block_d = 32
    
    grid = (
        triton.cdiv(sequence_length, block_m), # 第 0 维大小：Query 块的总数
        triton.cdiv(sequence_length, block_n), # 第 1 维大小：Key 块的总数
        batch_size * num_heads, # 第 2 维大小：(Batch * Head) 的总任务数
    )

    _attention_scores_kernel[grid](
        q,
        k,
        scores,
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
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        scores.stride(3),
        1.0 / math.sqrt(head_dim),
        CAUSAL=causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return scores


"""
    _attention_softmax_kernel 启动函数
"""
def attention_softmax(scores: torch.Tensor) -> torch.Tensor:
    """沿分数矩阵最后一维执行 Triton 逐行 Softmax，并显式写出 P。"""
    if not isinstance(scores, torch.Tensor):
        raise TypeError("scores 必须是 torch.Tensor")
    if not scores.is_cuda:
        raise ValueError("scores 必须位于 CUDA 设备")
    if scores.ndim != 4:
        raise ValueError("scores 必须是四维张量 [B, H, N, N]")
    if scores.dtype not in SUPPORTED_DTYPES:
        raise TypeError(
            f"不支持 dtype={scores.dtype}，仅支持 FP16、BF16 和 FP32"
        )
    if scores.shape[-2] != scores.shape[-1]:
        raise ValueError("scores 的最后两个维度必须相等，形状应为 [B, H, N, N]")
    if scores.stride(-1) != 1:
        raise ValueError("scores 的最后一维必须连续，Softmax 才能逐行加载")

    batch_size, num_heads, sequence_length, _ = scores.shape
    probabilities = torch.empty_like(scores)
    if scores.numel() == 0:
        return probabilities

    block_size = triton.next_power_of_2(sequence_length)
    if block_size > MAX_SOFTMAX_SIZE:
        raise ValueError(
            f"序列长度 {sequence_length} 超过单个 Softmax program 的支持范围；"
            "后续需要改用分块归约"
        )

    grid = (batch_size * num_heads * sequence_length,)
    _attention_softmax_kernel[grid](
        scores,
        probabilities,
        sequence_length,
        num_heads,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        scores.stride(3),
        probabilities.stride(0),
        probabilities.stride(1),
        probabilities.stride(2),
        probabilities.stride(3),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return probabilities


"""
    _attention_output_kernel 启动函数
"""
def attention_output(
    probabilities: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """使用 Triton 显式计算 O = P @ V。"""
    for name, tensor in (("probabilities", probabilities), ("v", v)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} 必须是 torch.Tensor")
        if not tensor.is_cuda:
            raise ValueError(f"{name} 必须位于 CUDA 设备")
        if tensor.ndim != 4:
            raise ValueError(f"{name} 必须是四维张量")

    if probabilities.dtype not in SUPPORTED_DTYPES:
        raise TypeError(
            f"不支持 dtype={probabilities.dtype}，仅支持 FP16、BF16 和 FP32"
        )
    if probabilities.dtype != v.dtype:
        raise TypeError("probabilities 和 v 的 dtype 必须一致")
    if probabilities.device != v.device:
        raise ValueError("probabilities 和 v 必须位于同一 CUDA 设备")

    batch_size, num_heads, query_length, key_length = probabilities.shape
    if query_length != key_length:
        raise ValueError("probabilities 的最后两个维度必须相等，形状应为 [B,H,N,N]")
    if v.shape[:3] != (batch_size, num_heads, key_length):
        raise ValueError(
            "v 的前三个维度必须与 probabilities 的 [B,H,N] 一致"
        )

    head_dim = v.shape[-1]
    output = torch.empty(
        (batch_size, num_heads, query_length, head_dim),
        device=v.device,
        dtype=v.dtype,
    )
    if output.numel() == 0:
        return output

    block_m = 32
    block_d = 32
    block_k = 32
    grid = (
        triton.cdiv(query_length, block_m),
        triton.cdiv(head_dim, block_d),
        batch_size * num_heads,
    )
    _attention_output_kernel[grid](
        probabilities,
        v,
        output,
        query_length,
        head_dim,
        num_heads,
        probabilities.stride(0),
        probabilities.stride(1),
        probabilities.stride(2),
        probabilities.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output


"""
    naive_attention 输入检查
"""
def _validate_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[int, int, int, int]:
    """检查朴素注意力当前支持的最小输入契约。"""
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} 必须是 torch.Tensor")
        if not tensor.is_cuda:
            raise ValueError(f"{name} 必须位于 CUDA 设备")
        if tensor.ndim != 4:
            raise ValueError(f"{name} 必须是四维张量 [B, H, N, D]")

    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q、k、v 的形状必须完全一致，均为 [B, H, N, D]")
    if q.dtype not in SUPPORTED_DTYPES:
        raise TypeError(
            f"不支持 dtype={q.dtype}，仅支持 FP16、BF16 和 FP32"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q、k、v 的 dtype 必须一致")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q、k、v 必须位于同一 CUDA 设备")

    batch_size, num_heads, sequence_length, head_dim = q.shape
    if head_dim == 0:
        raise ValueError("head_dim 必须大于 0，才能计算缩放因子 1/sqrt(D)")

    return batch_size, num_heads, sequence_length, head_dim


"""
    完整版 naive_attention
"""
def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    return_intermediates: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """串联三个 Triton Kernel，并显式物化分数矩阵 S 和概率矩阵 P。"""
    _validate_attention_inputs(q, k, v)
    if not isinstance(causal, bool):
        raise TypeError("causal 必须是 bool")
    if not isinstance(return_intermediates, bool):
        raise TypeError("return_intermediates 必须是 bool")

    # 生命周期：Q/K -> S 写入显存 -> P 写入显存 -> P/V -> O。
    scores = attention_scores(q, k, causal=causal)
    probabilities = attention_softmax(scores)
    output = attention_output(probabilities, v)

    if return_intermediates:
        return output, scores, probabilities
    return output


"""
    和 torch 版做比对
"""
def naive_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """显式物化分数矩阵和概率矩阵的 PyTorch 正确性参考。"""
    _, _, sequence_length, head_dim = _validate_attention_inputs(q, k, v)
    if not isinstance(causal, bool):
        raise TypeError("causal 必须是 bool")

    # [B,H,N,D] @ [B,H,D,N] -> [B,H,N,N]
    # scores 会真实占用全局显存，用来观察朴素注意力的 N^2 中间张量代价
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        # 主对角线上方满足 j > i，表示当前查询不允许读取的未来键位置。
        causal_mask = torch.triu(
            torch.ones(
                (sequence_length, sequence_length),
                dtype=torch.bool,
                device=q.device,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    # 最后一维是键位置。固定一个查询后，所有可见键的概率和应为 1
    probabilities = torch.softmax(scores, dim=-1)

    # [B,H,N,N] @ [B,H,N,D] -> [B,H,N,D]
    # probabilities 同样被显式保存，作为第二个 N^2 中间张量
    output = torch.matmul(probabilities, v)
    return output, scores, probabilities
