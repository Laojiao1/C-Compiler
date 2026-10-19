import torch
from transformers import GPT2LMHeadModel

from model import GPT2ReferenceModel


def test_gpt2_logits_alignment() -> None:
    # 参考模型只负责提供可信输出，不参与训练。
    # 当前阶段使用 float32，先排除 float16 舍入误差的干扰。
    reference_model = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        torch_dtype=torch.float32,
    )
    reference_model.eval()

    # 使用同一份 config 创建手写模型，保证层数、hidden size、
    # head 数量和词表大小与官方模型一致。
    custom_model = GPT2ReferenceModel(reference_model.config)
    custom_model.eval()

    # 我们刻意使用 strict=False，先查看两边 state_dict 的差异。
    # 正常情况下，只有 Hugging Face 内部额外保存的 masked_bias 不会被使用。
    missing_keys, unexpected_keys = custom_model.load_state_dict(
        reference_model.state_dict(),
        strict=False,
    )

    print("missing keys:", missing_keys)
    print("unexpected keys:", unexpected_keys)

    # masked_bias 只是 Hugging Face Attention 内部的辅助 buffer，
    # 参考实现不需要它。causal mask 也被标记为 non-persistent buffer，
    # 因此不会出现在 missing_keys 或 unexpected_keys 中。
    assert all(key.endswith("masked_bias") for key in unexpected_keys)
    assert not missing_keys

    # 这里直接使用固定的 token id，避免 tokenizer、文本处理和 padding
    # 混入本次模型结构对齐实验。
    input_ids = torch.tensor(
        [
            [15496, 11, 995, 318, 257],
            [464, 3797, 318, 257, 1332],
        ],
        dtype=torch.long,
    )

    # 对齐实验不需要梯度。关闭 autograd 可以减少内存和额外计算。
    with torch.no_grad():
        expected_logits = reference_model(input_ids).logits
        actual_logits = custom_model(input_ids)

    print("expected shape:", tuple(expected_logits.shape))
    print("actual shape:", tuple(actual_logits.shape))

    # 同时查看最大误差和平均误差：
    # 最大误差用于发现局部异常，平均误差用于判断整体是否稳定。
    diff = (expected_logits - actual_logits).abs()
    print("max abs error:", diff.max().item())
    print("mean abs error:", diff.mean().item())

    # 两个模型都使用 FP32，但底层矩阵乘法的累加顺序可能不同，
    # 因此允许一个小的数值误差；如果误差明显更大，应回到逐层输出定位。
    torch.testing.assert_close(
        actual_logits,
        expected_logits,
        rtol=2e-4,
        atol=2e-4,
    )
