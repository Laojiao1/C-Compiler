import importlib.util
from functools import lru_cache
from pathlib import Path

import pytest
import torch

if __package__:
    from .kernel import (
        _should_skip_causal_future_tiles,
        flash_attention_forward,
        flash_attention_forward_configured,
    )
else:
    from kernel import (
        _should_skip_causal_future_tiles,
        flash_attention_forward,
        flash_attention_forward_configured,
    )


CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="需要 CUDA GPU",
)
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
SHAPES = (
    (1, 1, 1, 1),      # 极小序列与 D<16 padding
    (1, 2, 32, 32),    # 单个对齐 Key Tile
    (1, 2, 64, 32),    # 两个 Key Tile，覆盖在线合并
    (2, 3, 45, 40),    # 多 batch/head，且 N/D 非对齐
)
TUNING_CONFIGS = (
    (16, 32, 4),
    (32, 16, 4),
    (32, 32, 2),
    (32, 32, 4),
    (32, 32, 8),
    (32, 64, 2),
    (32, 64, 4),
    (64, 32, 2),
    (64, 32, 4),
    (64, 32, 8),
    (64, 64, 2),
    (64, 64, 4),
    (64, 64, 8),
)


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 2e-2, 2e-2
    if dtype == torch.bfloat16:
        return 5e-2, 5e-2
    return 2e-4, 2e-4


def _skip_unsupported_bfloat16(dtype: torch.dtype) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("当前 CUDA GPU 不支持 BF16")


@pytest.mark.parametrize(
    ("causal", "sequence_length", "expected"),
    (
        (False, 2048, False),
        (True, 511, False),
        (True, 512, True),
    ),
)
def test_causal_skip_threshold(
    causal: bool,
    sequence_length: int,
    expected: bool,
):
    """锁定正式入口的实验阈值，避免后续重构改变选择语义。"""
    assert _should_skip_causal_future_tiles(sequence_length, causal) is expected


@lru_cache(maxsize=1)
def _load_naive_attention_module():
    """按文件路径复用 05 的 reference 和 Triton Naive 实现。"""
    module_path = (
        Path(__file__).resolve().parents[1]
        / "05_attention_naive"
        / "kernel.py"
    )
    spec = importlib.util.spec_from_file_location(
        "attention_naive_kernel_for_flash_test",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 05 实现：{module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@CUDA_REQUIRED
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("shape", SHAPES)
def test_flash_attention_matches_pytorch_reference(
    shape: tuple[int, int, int, int],
    causal: bool,
    dtype: torch.dtype,
):
    """覆盖 Shape、dtype、在线合并、边界 mask 和 causal 主路径。"""
    _skip_unsupported_bfloat16(dtype)
    torch.manual_seed(0)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)

    naive_module = _load_naive_attention_module()
    expected = naive_module.naive_attention_reference(
        q,
        k,
        v,
        causal=causal,
    )[0]
    actual = flash_attention_forward(q, k, v, causal=causal)

    assert actual.shape == shape
    assert actual.dtype == dtype
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@CUDA_REQUIRED
@pytest.mark.parametrize("causal", (False, True))
def test_flash_attention_matches_triton_naive(causal: bool):
    """代表 Shape 与 05 的三个 Triton Kernel 交叉验证。"""
    torch.manual_seed(1)
    shape = (1, 2, 64, 32)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)

    naive_module = _load_naive_attention_module()
    expected = naive_module.naive_attention(q, k, v, causal=causal)
    actual = flash_attention_forward(q, k, v, causal=causal)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@CUDA_REQUIRED
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize(("block_m", "block_n", "num_warps"), TUNING_CONFIGS)
def test_tuning_configs_match_reference(
    block_m: int,
    block_n: int,
    num_warps: int,
    causal: bool,
):
    """所有调优候选必须先通过同一个非对齐 correctness 用例。"""
    torch.manual_seed(2)
    shape = (1, 2, 65, 40)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)

    expected = _load_naive_attention_module().naive_attention_reference(
        q,
        k,
        v,
        causal=causal,
    )[0]
    actual = flash_attention_forward_configured(
        q,
        k,
        v,
        causal=causal,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@CUDA_REQUIRED
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
@pytest.mark.parametrize(
    "shape",
    (
        (1, 1, 1, 1),
        (1, 2, 45, 40),
        (2, 3, 129, 40),
    ),
)
def test_causal_future_tile_skipping_matches_reference(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
):
    """跳过完全未来 Tile 后仍需保持 causal Attention 语义。"""
    torch.manual_seed(3)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)

    expected = _load_naive_attention_module().naive_attention_reference(
        q,
        k,
        v,
        causal=True,
    )[0]
    actual = flash_attention_forward_configured(
        q,
        k,
        v,
        causal=True,
        block_m=64,
        block_n=32,
        num_warps=4,
        skip_causal_future_tiles=True,
    )

    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@CUDA_REQUIRED
def test_empty_sequence_returns_empty_output():
    q = torch.empty((2, 3, 0, 16), device="cuda", dtype=torch.float16)
    output = flash_attention_forward(q, q, q, causal=True)
    assert output.shape == q.shape
    assert output.numel() == 0


def test_rejects_non_tensor_input():
    tensor = torch.empty((1, 1, 1, 1))
    with pytest.raises(TypeError, match="q 必须是 torch.Tensor"):
        flash_attention_forward([1.0], tensor, tensor)


def test_rejects_cpu_tensor():
    q = torch.randn((1, 1, 4, 16))
    with pytest.raises(ValueError, match="q 必须位于 CUDA 设备"):
        flash_attention_forward(q, q, q)


@CUDA_REQUIRED
def test_rejects_non_4d_input():
    q = torch.randn((1, 4, 16), device="cuda")
    with pytest.raises(ValueError, match="q 必须是四维张量"):
        flash_attention_forward(q, q, q)


@CUDA_REQUIRED
def test_rejects_shape_mismatch():
    q = torch.randn((1, 2, 8, 16), device="cuda")
    k = torch.randn((1, 2, 7, 16), device="cuda")
    with pytest.raises(ValueError, match="形状必须完全一致"):
        flash_attention_forward(q, k, q)


@CUDA_REQUIRED
def test_rejects_dtype_mismatch():
    q = torch.randn((1, 2, 8, 16), device="cuda", dtype=torch.float16)
    k = q.to(torch.float32)
    with pytest.raises(TypeError, match="dtype 必须一致"):
        flash_attention_forward(q, k, q)


@CUDA_REQUIRED
def test_rejects_unsupported_dtype():
    q = torch.ones((1, 2, 8, 16), device="cuda", dtype=torch.int32)
    with pytest.raises(TypeError, match="仅支持 FP16、BF16 和 FP32"):
        flash_attention_forward(q, q, q)


@CUDA_REQUIRED
def test_rejects_noncontiguous_last_dimension():
    q = torch.randn((1, 2, 8, 32), device="cuda")[..., ::2]
    assert q.stride(-1) != 1
    with pytest.raises(ValueError, match="最后一维必须连续"):
        flash_attention_forward(q, q, q)


@CUDA_REQUIRED
@pytest.mark.parametrize("head_dim", (0, 129))
def test_rejects_invalid_head_dimension(head_dim: int):
    q = torch.empty((1, 2, 8, head_dim), device="cuda")
    with pytest.raises(ValueError, match="head_dim 位于"):
        flash_attention_forward(q, q, q)


@CUDA_REQUIRED
def test_rejects_non_boolean_causal():
    q = torch.randn((1, 2, 8, 16), device="cuda")
    with pytest.raises(TypeError, match="causal 必须是 bool"):
        flash_attention_forward(q, q, q, causal=1)


@CUDA_REQUIRED
@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"block_m": 8}, "block_m 必须是"),
        ({"block_n": 128}, "block_n 必须是"),
        ({"num_warps": 1}, "num_warps 必须是"),
    ),
)
def test_rejects_invalid_tuning_config(kwargs: dict, message: str):
    q = torch.randn((1, 2, 8, 16), device="cuda")
    with pytest.raises(ValueError, match=message):
        flash_attention_forward_configured(q, q, q, **kwargs)


@CUDA_REQUIRED
def test_rejects_causal_skipping_for_non_causal_attention():
    q = torch.randn((1, 2, 8, 16), device="cuda")
    with pytest.raises(ValueError, match="只有 causal=True"):
        flash_attention_forward_configured(
            q,
            q,
            q,
            causal=False,
            skip_causal_future_tiles=True,
        )


@CUDA_REQUIRED
def test_rejects_non_boolean_causal_skipping_flag():
    q = torch.randn((1, 2, 8, 16), device="cuda")
    with pytest.raises(TypeError, match="skip_causal_future_tiles 必须是 bool"):
        flash_attention_forward_configured(
            q,
            q,
            q,
            causal=True,
            skip_causal_future_tiles=1,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
