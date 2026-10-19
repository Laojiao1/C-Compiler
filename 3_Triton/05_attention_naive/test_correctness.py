import math

import pytest
import torch
import torch.nn.functional as F

if __package__:
    from .kernel import (
        attention_output,
        attention_scores,
        attention_softmax,
        naive_attention,
        naive_attention_reference,
    )
else:
    # 兼容直接执行：python test_correctness.py
    from kernel import (
        attention_output,
        attention_scores,
        attention_softmax,
        naive_attention,
        naive_attention_reference,
    )


CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="需要 CUDA GPU",
)


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """低精度矩阵乘法和 Softmax 允许更大的浮点舍入误差。"""
    if dtype == torch.float16:
        return 2e-2, 2e-2
    if dtype == torch.bfloat16:
        return 5e-2, 5e-2
    return 1e-4, 1e-4


@CUDA_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 2, 32, 32),  # N/D 都与当前 32 大小的分块对齐
        (2, 4, 17, 8),   # N/D 都需要边界 mask
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_attention_scores_matches_pytorch(
    causal: bool,
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
):
    """单独验证第一个 Triton Kernel：QK^T、缩放、边界和因果遮罩。"""
    torch.manual_seed(0)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)

    actual = attention_scores(q, k, causal=causal)
    expected = torch.matmul(q, k.transpose(-2, -1))
    expected = expected * (1.0 / math.sqrt(shape[-1]))

    sequence_length = shape[-2]
    if causal:
        future_mask = torch.triu(
            torch.ones(
                (sequence_length, sequence_length),
                dtype=torch.bool,
                device=q.device,
            ),
            diagonal=1,
        )
        expected = expected.masked_fill(future_mask, float("-inf"))

    expected_shape = (
        shape[0],
        shape[1],
        sequence_length,
        sequence_length,
    )
    assert actual.shape == expected_shape
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@CUDA_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 2, 32, 32),  # Softmax 行宽与 BLOCK_SIZE 对齐
        (2, 4, 17, 8),   # Softmax 行宽补齐到 32，验证 padding 不污染归一化
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_attention_softmax_matches_pytorch_and_invariants(
    causal: bool,
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
):
    """单独验证第二个 Triton Kernel：逐行 Softmax、padding 和遮罩概率。"""
    torch.manual_seed(1)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)

    scores = attention_scores(q, k, causal=causal)
    actual = attention_softmax(scores)
    expected = torch.softmax(scores, dim=-1)

    assert actual.shape == scores.shape
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    # 固定一个 batch/head/query 后，所有键位置的概率和必须为 1。
    row_sums = actual.sum(dim=-1)
    torch.testing.assert_close(
        row_sums,
        torch.ones_like(row_sums),
        atol=atol,
        rtol=rtol,
    )

    if causal:
        sequence_length = shape[-2]
        future_mask = torch.triu(
            torch.ones(
                (sequence_length, sequence_length),
                dtype=torch.bool,
                device=q.device,
            ),
            diagonal=1,
        )
        torch.testing.assert_close(
            actual[..., future_mask],
            torch.zeros_like(actual[..., future_mask]),
            atol=0.0,
            rtol=0.0,
        )


@CUDA_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 2, 32, 32),  # N/D 与三个 MatMul 分块都对齐
        (2, 4, 17, 8),   # N/D 都需要 P@V 的边界 mask
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_attention_output_matches_pytorch(
    causal: bool,
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
):
    """单独验证第三个 Triton Kernel：O = P @ V 和 N/D 边界。"""
    torch.manual_seed(2)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)

    scores = attention_scores(q, k, causal=causal)
    probabilities = attention_softmax(scores)
    actual = attention_output(probabilities, v)
    expected = torch.matmul(probabilities, v)

    assert actual.shape == shape
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@CUDA_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 2, 32, 32),
        (2, 4, 17, 8),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_naive_attention_matches_reference_at_every_stage(
    causal: bool,
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
):
    """验证完整 Triton 链路，并逐层比较显式物化的 S、P 和 O。"""
    torch.manual_seed(3)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)

    actual_output, actual_scores, actual_probabilities = naive_attention(
        q,
        k,
        v,
        causal=causal,
        return_intermediates=True,
    )
    expected_output, expected_scores, expected_probabilities = (
        naive_attention_reference(q, k, v, causal=causal)
    )

    assert actual_output.shape == shape
    assert actual_scores.shape == expected_scores.shape
    assert actual_probabilities.shape == expected_probabilities.shape

    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(
        actual_scores,
        expected_scores,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        actual_probabilities,
        expected_probabilities,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        actual_output,
        expected_output,
        atol=atol,
        rtol=rtol,
    )


def test_rejects_non_tensor_inputs():
    tensor = torch.empty((1, 1, 1, 1))
    with pytest.raises(TypeError, match="q 必须是 torch.Tensor"):
        naive_attention([1.0], tensor, tensor)


def test_rejects_cpu_tensors():
    q = torch.randn((1, 1, 4, 8))
    with pytest.raises(ValueError, match="q 必须位于 CUDA 设备"):
        naive_attention(q, q, q)


@CUDA_REQUIRED
def test_rejects_non_4d_inputs():
    q = torch.randn((2, 4, 8), device="cuda")
    with pytest.raises(ValueError, match="q 必须是四维张量"):
        naive_attention(q, q, q)


@CUDA_REQUIRED
def test_rejects_qkv_shape_mismatch():
    q = torch.randn((1, 2, 8, 16), device="cuda")
    k = torch.randn((1, 2, 7, 16), device="cuda")
    v = torch.randn((1, 2, 8, 16), device="cuda")
    with pytest.raises(ValueError, match="形状必须完全一致"):
        naive_attention(q, k, v)


@CUDA_REQUIRED
def test_rejects_qkv_dtype_mismatch():
    q = torch.randn((1, 2, 8, 16), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 2, 8, 16), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((1, 2, 8, 16), device="cuda", dtype=torch.float16)
    with pytest.raises(TypeError, match="dtype 必须一致"):
        naive_attention(q, k, v)


@CUDA_REQUIRED
def test_rejects_unsupported_dtype():
    q = torch.ones((1, 2, 8, 16), device="cuda", dtype=torch.int32)
    with pytest.raises(TypeError, match="仅支持 FP16、BF16 和 FP32"):
        naive_attention(q, q, q)


@CUDA_REQUIRED
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"causal": 1}, "causal 必须是 bool"),
        ({"return_intermediates": 1}, "return_intermediates 必须是 bool"),
    ],
)
def test_rejects_invalid_boolean_parameters(kwargs, message):
    q = torch.randn((1, 2, 8, 16), device="cuda")
    with pytest.raises(TypeError, match=message):
        naive_attention(q, q, q, **kwargs)


@CUDA_REQUIRED
def test_rejects_zero_head_dim():
    q = torch.empty((1, 2, 8, 0), device="cuda")
    with pytest.raises(ValueError, match="head_dim 必须大于 0"):
        naive_attention(q, q, q)


@CUDA_REQUIRED
def test_attention_softmax_rejects_non_square_scores():
    scores = torch.randn((1, 2, 8, 7), device="cuda")
    with pytest.raises(ValueError, match="最后两个维度必须相等"):
        attention_softmax(scores)


@CUDA_REQUIRED
def test_attention_softmax_rejects_noncontiguous_last_dimension():
    scores = torch.randn((1, 2, 8, 16), device="cuda")[..., ::2]
    assert scores.shape == (1, 2, 8, 8)
    assert scores.stride(-1) != 1
    with pytest.raises(ValueError, match="最后一维必须连续"):
        attention_softmax(scores)


@CUDA_REQUIRED
def test_attention_output_rejects_incompatible_value_shape():
    probabilities = torch.randn((1, 2, 8, 8), device="cuda")
    v = torch.randn((1, 2, 7, 16), device="cuda")
    with pytest.raises(ValueError, match="v 的前三个维度"):
        attention_output(probabilities, v)


@CUDA_REQUIRED
def test_empty_sequence_returns_empty_intermediates_and_output():
    q = torch.empty((2, 3, 0, 8), device="cuda")
    output, scores, probabilities = naive_attention(
        q,
        q,
        q,
        causal=True,
        return_intermediates=True,
    )

    assert output.shape == (2, 3, 0, 8)
    assert scores.shape == (2, 3, 0, 0)
    assert probabilities.shape == (2, 3, 0, 0)
    assert output.numel() == scores.numel() == probabilities.numel() == 0


@CUDA_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 2, 16, 8),  # 普通对齐序列长度
        (2, 4, 17, 8),  # 非 2 的幂序列长度
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_reference_matches_pytorch_and_attention_invariants(
    causal: bool,
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
):
    """验证对齐/非对齐形状、不同 dtype、概率归一化和因果遮罩。"""
    torch.manual_seed(0)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)

    output, scores, probabilities = naive_attention_reference(
        q,
        k,
        v,
        causal=causal,
    )
    expected = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=causal,
    )

    batch_size, num_heads, sequence_length, _ = shape
    expected_matrix_shape = (
        batch_size,
        num_heads,
        sequence_length,
        sequence_length,
    )
    assert scores.shape == expected_matrix_shape
    assert probabilities.shape == expected_matrix_shape
    assert output.shape == shape
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)

    # 每个查询都在最后一维的所有键之间分配概率，因此每行之和应为 1。
    row_sums = probabilities.sum(dim=-1)
    torch.testing.assert_close(
        row_sums,
        torch.ones_like(row_sums),
        atol=atol,
        rtol=rtol,
    )

    if causal:
        # 上三角满足 j > i：这些未来位置必须在 Softmax 前后分别为 -inf 和 0。
        future_mask = torch.triu(
            torch.ones(
                (sequence_length, sequence_length),
                dtype=torch.bool,
                device=q.device,
            ),
            diagonal=1,
        )
        assert torch.isneginf(scores[..., future_mask]).all()
        torch.testing.assert_close(
            probabilities[..., future_mask],
            torch.zeros_like(probabilities[..., future_mask]),
            atol=0.0,
            rtol=0.0,
        )


if __name__ == "__main__":
    # 保持项目既有习惯：直接运行 python test_correctness.py 也会执行测试。
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
