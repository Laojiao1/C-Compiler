# Triton 写 KV + FlashAttention 前向
import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


"""
每个 token 的 KV 存到 KV Cache 里它该在的槽位
"""
# 为什么要专门写个 Triton kernel，而不用 PyTorch 的高级索引 k_cache[slot_mapping] = key？
# 因为 KV Cache 是分块的，slot 是离散跳跃的，Pytorch 的 gather/scatter 会产生额外的中间张量和访存。一个定制 kernel 直接「一个线程写一个 token」，最省
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)

    if slot == -1: return # -1 表示这个 token 不需要写（padding）

    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    # 用传入的 stride 定位第 idx 个 token 的 D 个元素，写到 cache 的 slot 行
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    cache_offsets = slot * D + tl.arange(0, D) # 算出在全局 KV Cache 中的精确物理字节偏移

    # 原地写入显存池
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


# Attention 在这里既是「写入方」也是「读取方」
class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    """
     prefill 用 flash_attn_varlen_func：一次要算整段 prompt 的注意力，多条序列变长地拼在一起（靠 cu_seqlens 累积长度切分），causal=True 保证每个位置只看前文。当本步命中了前缀缓存（block_tables is not None），k, v 会被整块替换为 k_cache/v_cache，再借 block_table 索引物理块，把已缓存的历史 K/V 一并纳入注意力。

     decode 用 flash_attn_with_kvcache：每条序列只新增一个 token（q.unsqueeze(1) 补出长度维 1），K/V 全部来自 Cache，用 cache_seqlens 告诉 kernel 每条序列当前多长、block_table 告诉它去哪些物理块里读。
    """
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel(): # KV cache 已分配，才写
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            # 命中前缀缓存：改从整块 cache 读历史 KV
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
