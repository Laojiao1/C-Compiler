"""5.3.1 单 Query、单层 CPU/FP32 Attention reference，支持 MHA/GQA/MQA。"""

import math

import torch


@torch.no_grad()
def single_query_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """输入 q:[Hq,D]，k/v:[T,Hkv,D]；返回 [Hq,D]。

    当前 Query 可见传入的全部 T 个位置，不再添加 causal mask。
    T 必须大于 0，Hq 必须能被 Hkv 整除。无 dropout。
    """
    if q.ndim != 2 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q 必须为 [Hq,D]，k/v 必须为 [T,Hkv,D]")
    if k.shape != v.shape or q.shape[1] != k.shape[2]:
        raise ValueError("K/V shape 必须相同，Q/K/V 的 head_dim 必须一致")
    num_query_heads, head_dim = q.shape
    num_tokens, num_kv_heads, _ = k.shape
    if min(num_query_heads, num_tokens, num_kv_heads, head_dim) <= 0:
        raise ValueError("头数、有效历史长度和 head_dim 必须大于 0")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("Query heads 必须能被 KV heads 整除")
    for tensor in (q, k, v):
        if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
            raise ValueError("本步只接收 CPU 上的 float32 Q/K/V")

    group_size = num_query_heads // num_kv_heads
    outputs = []

    for query_head in range(num_query_heads):
        kv_head = query_head // group_size

        scores = k[:, kv_head, :] @ q[query_head]
        scores = scores / math.sqrt(head_dim)

        weights = torch.softmax(scores, dim=0)

        head_output = weights @ v[:, kv_head, :]
        outputs.append(head_output)

    return torch.stack(outputs, dim=0)
