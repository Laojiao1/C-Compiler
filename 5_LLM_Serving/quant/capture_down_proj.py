"""
用前向 hook 从同一层 `mlp.down_proj` 抓两小组真实输入，分别作校准和验证。取出对应权重，记录层号、形状和 dtype。把 batch 和 sequence 维展平，令 `X: [T,K]`、`W: [N,K]`。"""
import argparse
from pathlib import Path

import torch
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LAYER_INDEX = 10
TEXTS = {
    "calibration": "量化时，一组权重共享一个缩放因子。分组大小决定了这个因子覆盖的范围。",
    "validation": "推理时，解码阶段逐步生成新词元。注意力层会读取前面词元的键和值。",
}


def capture_linear_inputs(tokenizer, model):
    linear = model.model.layers[LAYER_INDEX].mlp.down_proj
    weight = linear.weight.detach().float()
    bias = linear.bias.detach().float() if linear.bias is not None else None
    samples = {}

    def capture(module, args, output):
        # Linear 的最后一维是输入通道。把前面的 batch、sequence 合并成 token 行。
        activations = args[0].detach().flatten(0, -2).float()
        layer_output = output.detach().flatten(0, -2).float()
        samples[split] = {"x": activations, "layer_output": layer_output}

    handle = linear.register_forward_hook(capture)
    try:
        for split, text in TEXTS.items():
            encoded = tokenizer(text, return_tensors="pt").to(model.device)
            model.model(**encoded, use_cache=False)
    finally:
        handle.remove()

    for split, sample in samples.items():
        # hook 取到的是 FP16 模型的输入。转 FP32 后重算这一层作为误差基准。
        x = sample["x"]
        reference = x @ weight.T
        if bias is not None:
            reference += bias
        layer_output = sample.pop("layer_output")
        relative_error = (reference - layer_output).norm() / reference.norm()
        print(f"{split}: X={tuple(x.shape)}, Y={tuple(reference.shape)}, "
              f"FP16 layer vs FP32 reference relerr={relative_error.item():.3e}")
        sample["y_ref"] = reference.cpu()
        sample["x"] = x.cpu()

    print(f"layer={LAYER_INDEX}, W={tuple(weight.shape)}, model dtype={linear.weight.dtype}")
    return {
        "model_id": MODEL_ID,
        "revision": getattr(model.config, "_commit_hash", None),
        "layer_index": LAYER_INDEX,
        "model_dtype": str(linear.weight.dtype),
        "texts": TEXTS,
        "weight": weight.cpu(),
        "bias": bias.cpu() if bias is not None else None,
        **samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("down_proj_sample.pt"))
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Cached Hugging Face revision")
    args = parser.parse_args()

    config_path = try_to_load_from_cache(MODEL_ID, "config.json", revision=args.revision)
    if not isinstance(config_path, str):
        parser.error(f"No cached config for {MODEL_ID} at revision {args.revision}")

    print(f"PyTorch {torch.__version__}, GPU {torch.cuda.get_device_name()}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=args.revision, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=args.revision, dtype=torch.float16, local_files_only=True
    ).to("cuda").eval()
    with torch.inference_mode():
        data = capture_linear_inputs(tokenizer, model)
    torch.save(data, args.output)
    print(f"revision={data['revision']}, saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
