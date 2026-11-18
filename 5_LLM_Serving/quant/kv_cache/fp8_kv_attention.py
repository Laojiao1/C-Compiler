import torch
from transformers.models.qwen2.modeling_qwen2 import rotate_half

from .fp8_kv_cache import FP8KVCache, LAYER_INDEX, VALIDATION_TEXT, load_qwen, read_layer_cache


def capture_decode_inputs():
    model, tokenizer = load_qwen()
    attention = model.model.layers[LAYER_INDEX].self_attn
    captured = {}

    def capture_rope(_module, _args, kwargs):
        cos, sin = kwargs["position_embeddings"]
        captured["cos"] = cos[0, -1]
        captured["sin"] = sin[0, -1]

    def capture_query(_module, _args, output):
        captured["query"] = output[0, -1].reshape(-1, attention.head_dim)

    rope_handle = attention.register_forward_pre_hook(capture_rope, with_kwargs=True)
    query_handle = attention.q_proj.register_forward_hook(capture_query)
    try:
        k, v = read_layer_cache(model, tokenizer, VALIDATION_TEXT)
    finally:
        rope_handle.remove()
        query_handle.remove()

    # 只取最后一个 token 的 Q，并复用模型传给该层的 RoPE 系数。
    # K 已经过 RoPE，直接从模型缓存读取。Q 和缓存属于同一次前向。
    query = captured["query"]
    query = query * captured["cos"] + rotate_half(query) * captured["sin"]
    return query, k, v


def decode_attention(
    query: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # GQA：每连续 6 个 Q head 对应 1 个 KV head
    # 将 Q reshape 成 [kv_head, queries_per_kv_head, head_dim]，这样就无需复制缓存
    _, kv_heads, head_dim = k.shape 
    q = query.float().reshape(kv_heads, -1, head_dim) # [12, 128] -> [2, 6, 128]
    # kv.shape = [tokens, kv_head, head_dim]
    keys = k.float().permute(1, 2, 0)  # [31, 2, 128] -> [2, 128, 31] [kv_head, head_dim, token]
    values = v.float().transpose(0, 1)  # [31, 2, 128] -> [2, 31, 128] [kv_head, token, head_dim]

    # Q 来自序列的最后一个位置，可读缓存中所有有效 token，包括自己。
    logits = (q @ keys) * (head_dim ** -0.5) # [2, 6, 128] @ [2, 128, 31] -> [2, 6, 31]
    probabilities = logits.softmax(dim=-1)

    return probabilities, probabilities @ values # [2, 6, 31] @ [2, 31, 128] -> [2, 6, 128]


@torch.inference_mode()
def main() -> None:
    query, k, v = capture_decode_inputs()
    cache = FP8KVCache(k, k.shape[0])
    cache.append(k[:-1], v[:-1])
    cache.append(k[-1:], v[-1:])

    # 读取实际写进缓存的 FP8 字节及对应 scale，反量化到 FP32 供参考计算。
    k_fp8 = cache.k.float() * cache.k_scale.float()[..., None]
    v_fp8 = cache.v.float() * cache.v_scale.float()[..., None]

    p_ref, out_ref = decode_attention(query, k, v)
    print(f"Layer {LAYER_INDEX}: Q={tuple(query.shape)}, KV={tuple(k.shape)}")
    print(f"{'case':<14} {'mean TV':>12} {'output relerr':>16}")
    for label, keys, values in (
        ("FP16 K/V", k, v),
        ("FP8 K only", k_fp8, v),
        ("FP8 V only", k, v_fp8),
        ("FP8 K/V", k_fp8, v_fp8),
    ):
        p, out = decode_attention(query, keys, values)
        # 对每个 Q head 算 0.5 × 概率绝对差之和，再对所有 Q heads 取平均。
        tv = (0.5 * (p - p_ref).abs().sum(dim=-1)).mean().item()
        relerr = ((out - out_ref).norm() / out_ref.norm()).item()
        if label == "FP8 V only":
            torch.testing.assert_close(p, p_ref, rtol=0, atol=0)
        print(f"{label:<14} {tv:12.4%} {relerr:16.4%}")


if __name__ == "__main__":
    main()
