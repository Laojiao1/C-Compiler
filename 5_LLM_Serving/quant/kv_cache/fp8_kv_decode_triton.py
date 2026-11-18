import torch
import triton
import triton.language as tl

from .fp8_kv_attention import capture_decode_inputs, decode_attention
from .fp8_kv_cache import FP8KVCache

@triton.jit
def fp8_decode_kernel(
    query_ptr, k_ptr, v_ptr, k_scale_ptr, v_scale_ptr, out_ptr,
    length,
    KV_HEADS: tl.constexpr, QUERIES_PER_KV_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_T: tl.constexpr, DOT_SCALE: tl.constexpr
):
    q_head = tl.program_id(0) # 启动网格为 q_head = 12
    kv_head = q_head // QUERIES_PER_KV_HEAD # 0 / 1
    offset_dim = tl.arange(0, HEAD_DIM)
    q = tl.load(query_ptr + q_head * HEAD_DIM + offset_dim).to(tl.float32)

    running_max = -float("inf")
    normalizer = 0.0
    weighted_value = tl.full((HEAD_DIM,), 0, tl.float32)

    for start in range(0, length, BLOCK_T):
        tokens = start + tl.arange(0, BLOCK_T)
        mask = tokens < length
        # kv 布局为 [tokens, kv_head, head_dim], 
        # scale 布局为 [tokens, kv_head]
        scale_offsets = tokens * KV_HEADS + kv_head # 定位[tokens,  kv_head]
        data_offsets = scale_offsets[:, None] * HEAD_DIM + offset_dim[None, :]

        # 加载 scale
        k_scale = tl.load(k_scale_ptr + scale_offsets, mask=mask, other=0).to(tl.float32)
        v_scale = tl.load(v_scale_ptr + scale_offsets, mask=mask, other=0).to(tl.float32)

        # 注意这里，全局显存（HBM）把当前 Tile 的 FP8 字节和 FP16 Scale 读进寄存器后，立刻在寄存器里完成 bitcast -> float32 -> * scale 反量化，全过程不向全局显存写回任何中间 FP32/FP16 张量，节省带宽
        k_bits = tl.load(k_ptr + data_offsets, mask=mask[:, None], other=0)
        v_bits = tl.load(v_ptr + data_offsets, mask=mask[:, None], other=0)

        k = k_bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        v = v_bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)

        k = k * k_scale[:, None]
        v = v * v_scale[:, None]

        logits = tl.sum(k * q[None, :], axis=1) * DOT_SCALE # 这里算完变成 1 维了
        logits = tl.where(mask, logits, -float("inf"))

        # online-softmax
        new_max = tl.maximum(running_max, tl.max(logits, axis=0))
        alpha = tl.exp(running_max - new_max)
        block_weights_exp = tl.exp(logits - new_max)
        normalizer = normalizer * alpha + tl.sum(block_weights_exp, axis=0)
        weighted_value = weighted_value * alpha + tl.sum(block_weights_exp[:, None] * v, axis=0)
        running_max = new_max

    output_offset = q_head * HEAD_DIM + offset_dim
    tl.store(out_ptr + output_offset, weighted_value / normalizer)


# decode 阶段，tokens = 1, 传进来的 query.shape = [q_heads, head_dim]
def decode_fp8(query: torch.Tensor, cache: FP8KVCache, length: int) -> torch.Tensor:
    # 本实验只实现逐 token、逐 KV head 定标，以及非空缓存。
    if cache.fixed or not 1 <= length <= cache.length:
        raise ValueError("Expected per-token-head scales and a nonempty valid prefix")
    
    kv_heads = cache.k.shape[1]
    q_heads, head_dim = query.shape
    output = torch.empty_like(query, dtype=torch.float32)
    fp8_decode_kernel[(q_heads,)](
        query, cache.k.view(torch.uint8), cache.v.view(torch.uint8),
        cache.k_scale, cache.v_scale, output, length,
        KV_HEADS = kv_heads,
        QUERIES_PER_KV_HEAD = q_heads // kv_heads,
        HEAD_DIM = head_dim,
        BLOCK_T = 16,
        DOT_SCALE = head_dim ** -0.5,
    )
    return output


def report_capacity(cache: FP8KVCache, fp16_bytes_per_value: int) -> None:
    k, v, k_scale, v_scale = cache.k, cache.v, cache.k_scale, cache.v_scale
    buffers = (k, v, k_scale, v_scale)
    capacity = k.shape[0]

    # 四个缓存张量独立持有存储。切片只用于计算有效前缀的逻辑字节数。
    allocated = [tensor.untyped_storage().nbytes() for tensor in buffers]
    occupied = [tensor[:cache.length].numel() * tensor.element_size() for tensor in buffers]

    print(f"KV cache capacity: {cache.length}/{capacity} tokens")
    print(f"{'buffer':<15} {'occupied (B)':>13} {'allocated (B)':>14}")
    for name, used, reserved in zip(("K", "V", "K scale", "V scale"), occupied, allocated):
        print(f"{name:<15} {used:13,d} {reserved:14,d}")
    print(f"{'FP8 total':<15} {sum(occupied):13,d} {sum(allocated):14,d}")

    # FP16 对照按同样的形状和预分配长度折算，只需存 K/V 两份数据。
    fp16_per_token = (k[0].numel() + v[0].numel()) * fp16_bytes_per_value
    fp8_per_token = sum(occupied) // cache.length
    
    print(f"{'FP16 estimate':<15} {cache.length * fp16_per_token:13,d} "
          f"{capacity * fp16_per_token:14,d}")
    print(f"Unused FP8 slots: {sum(allocated) - sum(occupied):,} B")
    print(f"One layer per token: FP16={fp16_per_token:,} B, FP8={fp8_per_token:,} B")


@torch.inference_mode()
def main() -> None:
    query, k, v = capture_decode_inputs()
    # 预留空间，确保 kernel 遍历有效长度，而不是缓冲区容量
    cache = FP8KVCache(k, k.shape[0] + 3) # 31+3 = 34
    cache.append(k[:-1], v[:-1]) # 模拟 prefill，取到倒数第2个
    cache.append(k[-1:], v[-1:]) # 模拟 decode，取最后一个

    for length in (1, 16, cache.length): # 依次传入 1，16，31
        actual = decode_fp8(query, cache, length)

        k_ref = cache.k[:length].float() * cache.k_scale[:length].float()[..., None] # [length, 2, 128] * [length, 2, 1]
        v_ref = cache.v[:length].float() * cache.v_scale[:length].float()[..., None]
        _, ref = decode_attention(query, k_ref, v_ref)
        ref = ref.reshape_as(actual)

        torch.testing.assert_close(actual, ref, rtol=1e-4, atol=1e-5)

        error = (actual - ref).abs().max().item()

        print(f"length={length}, capacity={cache.k.shape[0]}, "
                      f"output={tuple(actual.shape)}, max_abs_error={error:.6e}")

    report_capacity(cache, k.element_size())


if __name__ == "__main__":
    main()
