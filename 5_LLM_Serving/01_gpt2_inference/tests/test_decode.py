import torch
from transformers import GPT2LMHeadModel

from decode import decode_with_cache, decode_without_cache
from model import GPT2ReferenceModel


def test_three_decode_paths_match() -> None:
    """三条路径应生成相同 token，且每一步 logits 都在误差范围内一致。"""
    reference_model = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        torch_dtype=torch.float32,
    )
    reference_model.eval()

    model = GPT2ReferenceModel(reference_model.config)
    model.load_state_dict(reference_model.state_dict(), strict=False)
    model.eval()

    input_ids = torch.tensor(
        [[15496, 11, 995, 318, 257]],
        dtype=torch.long,
    )
    max_new_tokens = 4

    # 路径 A：每一步重新计算完整序列。
    full_ids, full_logits = decode_without_cache(
        model,
        input_ids,
        max_new_tokens,
    )
    # 路径 B：使用 torch.cat 追加历史 K/V。
    cat_ids, cat_logits = decode_with_cache(
        model,
        input_ids,
        max_new_tokens,
        mode="cat",
    )
    # 路径 C：使用预分配 Cache，当前位置原地写入。
    prealloc_ids, prealloc_logits = decode_with_cache(
        model,
        input_ids,
        max_new_tokens,
        mode="prealloc",
    )

    assert torch.equal(full_ids, cat_ids)
    assert torch.equal(full_ids, prealloc_ids)

    assert len(full_logits) == max_new_tokens
    assert len(cat_logits) == max_new_tokens
    assert len(prealloc_logits) == max_new_tokens

    for expected, actual in zip(full_logits, cat_logits):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)

    for expected, actual in zip(full_logits, prealloc_logits):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
