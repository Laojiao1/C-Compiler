import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from weight_quant.capture_down_proj import DEFAULT_REVISION, LAYER_INDEX, MODEL_ID


FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max
CALIBRATION_TEXT = "矩阵乘法把每一行输入投影到输出空间，注意力层则根据当前查询从历史序列中读取信息。"
VALIDATION_TEXT = "图书馆周末开放自习室。读者可以在入口登记，也可以在网上提前预约。管理员每天检查书架，并把新到的图书放到展示区。"


def fp16_scale(values: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    amax = values.float().abs().amax(dim=dim)
    desired = torch.where(amax > 0, amax / FP8_MAX, 1.0)
    stored = desired.half()
    upper = torch.nextafter(stored, torch.full_like(stored, float("inf"))) # 防截断
    return torch.where(stored.float() < desired, upper, stored)


def encode_fp8(values: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, int]:
    # 先除以缓存中实际保存的 scale，再显式饱和，避免 E4M3 溢出产生 NaN。
    normalized = values.float() / scale.float()[..., None]
    saturated = int((normalized.abs() > FP8_MAX).sum().item())
    return normalized.clamp(-FP8_MAX, FP8_MAX).to(FP8), saturated


class FP8KVCache:
    def __init__(
        self,
        template: torch.Tensor,
        capacity: int,
        fixed_scales: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        # [token, kv_head, head_dim]，只存 KV heads。分配后追加通过切片写入。
        _, heads, width = template.shape
        shape = (capacity, heads, width)
        self.k = template.new_empty(shape, dtype=FP8)
        self.v = template.new_empty(shape, dtype=FP8)
        self.length = 0
        self.fixed = fixed_scales is not None
        if fixed_scales is None:
            self.k_scale = template.new_empty((capacity, heads), dtype=torch.float16)
            self.v_scale = template.new_empty((capacity, heads), dtype=torch.float16)
        else:
            self.k_scale, self.v_scale = fixed_scales

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[int, int]:
        start = self.length
        end = start + k.shape[0]
        if end > self.k.shape[0]:
            raise ValueError("KV Cache capacity exceeded")

        k_scale = self.k_scale if self.fixed else fp16_scale(k, dim=-1)
        v_scale = self.v_scale if self.fixed else fp16_scale(v, dim=-1)
        k_codes, k_saturated = encode_fp8(k, k_scale)
        v_codes, v_saturated = encode_fp8(v, v_scale)
        self.k[start:end].copy_(k_codes)
        self.v[start:end].copy_(v_codes)
        if not self.fixed:
            self.k_scale[start:end].copy_(k_scale)
            self.v_scale[start:end].copy_(v_scale)
        self.length = end
        return k_saturated, v_saturated

    def prefix_bytes(self, length: int) -> tuple[torch.Tensor, ...]:
        # 固定 scale 也是缓存的一部分。把它和旧条目的数据一起核对。
        k_scale = self.k_scale if self.fixed else self.k_scale[:length]
        v_scale = self.v_scale if self.fixed else self.v_scale[:length]
        return tuple(tensor.view(torch.uint8) for tensor in
                     (self.k[:length], self.v[:length], k_scale.reshape(-1), v_scale.reshape(-1)))


def read_layer_cache(model, tokenizer, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt").to(model.device)
    output = model.model(**encoded, use_cache=True)
    layer = output.past_key_values.layers[LAYER_INDEX]
    # Hugging Face 缓存中的 K 已经过 RoPE，V 是投影后的值。
    return (
        layer.keys[0].transpose(0, 1).contiguous(),
        layer.values[0].transpose(0, 1).contiguous(),
    )


def load_qwen():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=DEFAULT_REVISION, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=DEFAULT_REVISION, dtype=torch.float16, local_files_only=True
    ).to("cuda").eval()
    return model, tokenizer


@torch.inference_mode()
def main() -> None:
    model, tokenizer = load_qwen()
    k_cal, v_cal = read_layer_cache(model, tokenizer, CALIBRATION_TEXT)
    k, v = read_layer_cache(model, tokenizer, VALIDATION_TEXT)
    fixed_scales = (fp16_scale(k_cal), fp16_scale(v_cal))
    print(f"Layer {LAYER_INDEX}, KV shape={tuple(k.shape)}, calibration tokens={k_cal.shape[0]}")
    print(f"Fixed K/V scales: {fixed_scales[0].item():.6g}, {fixed_scales[1].item():.6g}")

    # 固定使用真实序列的前缀及最后三个 token，再单独追加一个人为越界条目。
    prefix = k.shape[0] - 3
    stress_k, stress_v = k[-1:].clone(), v[-1:].clone()
    # 注入离群值
    stress_k[0, 0, 0] = 2 * FP8_MAX * fixed_scales[0].float()
    stress_v[0, 0, 0] = 2 * FP8_MAX * fixed_scales[1].float()
    for name, scales in (("fixed", fixed_scales), ("per-token-head", None)):
        cache = FP8KVCache(k, k.shape[0] + 1, scales)
        cache.append(k[:prefix], v[:prefix])
        before = tuple(tensor.clone() for tensor in cache.prefix_bytes(prefix))
        for t in range(prefix, k.shape[0]):
            print(f"{name}: real token {t}, saturated K/V={cache.append(k[t:t+1], v[t:t+1])}")
        stress = cache.append(stress_k, stress_v)
        assert all(torch.equal(old, new) for old, new in zip(before, cache.prefix_bytes(prefix)))
        print(f"{name}: stress token, saturated K/V={stress}, old prefix bytes unchanged, length={cache.length}")
        byte_count = sum(t.numel() * t.element_size() for t in (cache.k, cache.v))
        scale_bytes = sum(t.numel() * t.element_size() for t in (cache.k_scale, cache.v_scale))
        print(f"{name}: allocated payload={byte_count} B, scales={scale_bytes} B")


if __name__ == "__main__":
    main()
