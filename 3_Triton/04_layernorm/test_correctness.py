import pytest
import torch
import torch.nn.functional as F

if __package__:
    from .kernel import MAX_FUSED_SIZE, layer_norm, layer_norm_autotuned
else:
    # 兼容直接执行：python test_correctness.py
    from kernel import MAX_FUSED_SIZE, layer_norm, layer_norm_autotuned


CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="需要 CUDA GPU",
)


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """低精度输入允许更大的 reduction 舍入误差。"""
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
        (1, 1),       # 最小非空输入
        (3, 5),       # 大部分 BLOCK_SIZE 位置都是 padding
        (123, 555),   # 非 2 的幂特征维度，重点验证 mask
        (64, 512),    # 对齐的中等特征维度
        (256, 1024),  # 常见的 2 的幂特征维度
    ],
)
# 测试数学正确性
def test_layer_norm_matches_pytorch(shape, dtype):
    torch.manual_seed(0)
    _, n_cols = shape
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(n_cols, device="cuda", dtype=dtype)
    bias = torch.randn(n_cols, device="cuda", dtype=dtype)

    expected = F.layer_norm(x, (n_cols,), weight, bias, eps=1e-5)
    manual_actual = layer_norm(x, weight, bias, eps=1e-5)

    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(manual_actual, expected, atol=atol, rtol=rtol)


@CUDA_REQUIRED
@pytest.mark.parametrize(
    "shape",
    [
        (3, 5),        # 小 BLOCK_SIZE
        (64, 512),     # 中等对齐特征维度
        (32, 2049),    # BLOCK_SIZE 翻倍且包含大量 padding
        (128, 32768),  # 极长 reduction，实测 warp 配置差异最大
    ],
)
# 测试 autotuner 选择的配置不会改变数学结果
def test_autotuned_layer_norm_matches_pytorch(shape):
    """autotuner 只改变执行配置，不能改变 LayerNorm 数学结果。"""
    torch.manual_seed(1)
    _, n_cols = shape
    x = torch.randn(shape, device="cuda", dtype=torch.float16)
    weight = torch.randn(n_cols, device="cuda", dtype=torch.float16)
    bias = torch.randn(n_cols, device="cuda", dtype=torch.float16)

    expected = F.layer_norm(x, (n_cols,), weight, bias, eps=1e-5)
    actual = layer_norm_autotuned(x, weight, bias, eps=1e-5)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

# 测试零方差情况
@CUDA_REQUIRED
def test_constant_rows_reduce_to_bias(): 
    """零方差时 normalized 为 0，所以 affine 结果必须等于 bias。"""
    n_rows, n_cols = 10, 127
    x = torch.full((n_rows, n_cols), 3.25, device="cuda", dtype=torch.float32)
    weight = torch.randn(n_cols, device="cuda", dtype=torch.float32)
    bias = torch.randn(n_cols, device="cuda", dtype=torch.float32)

    actual = layer_norm(x, weight, bias)
    expected = F.layer_norm(x, (n_cols,), weight, bias, eps=1e-5)
    theoretical = bias.unsqueeze(0).expand_as(x)

    # reduction、除法和 rsqrt 的硬件路径不保证逐 bit 得到理论上的 bias。
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual, theoretical, atol=2e-5, rtol=2e-5)
    assert torch.isfinite(actual).all()

# 测试大偏置加小波动的数值稳定性
@CUDA_REQUIRED
def test_large_offset_small_variance_is_stable():
    """大公共偏置加小波动会放大 E[x^2]-E[x]^2 的消减误差。"""
    torch.manual_seed(1)
    n_rows, n_cols = 32, 2049
    x = 10000.0 + 0.25 * torch.randn(
        (n_rows, n_cols),
        device="cuda",
        dtype=torch.float32,
    )
    weight = torch.randn(n_cols, device="cuda", dtype=torch.float32)
    bias = torch.randn(n_cols, device="cuda", dtype=torch.float32)

    expected = F.layer_norm(x, (n_cols,), weight, bias, eps=1e-5)
    actual = layer_norm(x, weight, bias, eps=1e-5)

    # 参考值接近 0 时，相对误差会被显著放大，因此以绝对误差为主要边界。
    # 当前两遍中心化算法仍受 FP32 mean reduction 舍入误差影响；Welford 留作后续实验。
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-2)
    assert torch.isfinite(actual).all()

# 测试空输入的边界情况
@CUDA_REQUIRED
@pytest.mark.parametrize("shape", [(0, 128), (4, 0)])
def test_empty_input_returns_empty_output(shape):
    _, n_cols = shape
    x = torch.empty(shape, device="cuda", dtype=torch.float32)
    weight = torch.empty(n_cols, device="cuda", dtype=torch.float32)
    bias = torch.empty(n_cols, device="cuda", dtype=torch.float32)

    actual = layer_norm(x, weight, bias)

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert actual.device == x.device
    assert actual.numel() == 0

# ----- 测试输入验证逻辑 ----- 
def test_rejects_cpu_tensors():
    x = torch.randn(2, 8)
    weight = torch.randn(8)
    bias = torch.randn(8)

    with pytest.raises(ValueError, match="必须位于 CUDA 设备"):
        layer_norm(x, weight, bias)


@CUDA_REQUIRED
def test_rejects_noncontiguous_last_dimension():
    x = torch.randn(5, 7, device="cuda", dtype=torch.float32).T
    weight = torch.randn(x.shape[1], device="cuda", dtype=torch.float32)
    bias = torch.randn(x.shape[1], device="cuda", dtype=torch.float32)
    assert x.stride(1) != 1

    with pytest.raises(ValueError, match="最后一维必须连续"):
        layer_norm(x, weight, bias)


@CUDA_REQUIRED
def test_rejects_affine_length_mismatch():
    x = torch.randn(4, 16, device="cuda", dtype=torch.float32)
    weight = torch.randn(15, device="cuda", dtype=torch.float32)
    bias = torch.randn(16, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="长度必须等于"):
        layer_norm(x, weight, bias)


@CUDA_REQUIRED
def test_rejects_dtype_mismatch():
    x = torch.randn(4, 16, device="cuda", dtype=torch.float16)
    weight = torch.randn(16, device="cuda", dtype=torch.float32)
    bias = torch.randn(16, device="cuda", dtype=torch.float16)

    with pytest.raises(TypeError, match="dtype 必须一致"):
        layer_norm(x, weight, bias)


@CUDA_REQUIRED
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"eps": -1e-5}, "eps 必须是非负实数"),
        ({"num_warps": 3}, "num_warps 必须是 1、2、4 或 8"),
    ],
)
def test_rejects_invalid_scalar_parameters(kwargs, message):
    x = torch.randn(4, 16, device="cuda", dtype=torch.float32)
    weight = torch.randn(16, device="cuda", dtype=torch.float32)
    bias = torch.randn(16, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match=message):
        layer_norm(x, weight, bias, **kwargs)


# 测试特征维度超过单个 program 支持范围的情况
@CUDA_REQUIRED 
def test_rejects_feature_dimension_above_single_program_limit():
    n_cols = MAX_FUSED_SIZE + 1
    x = torch.randn(1, n_cols, device="cuda", dtype=torch.float32)
    weight = torch.randn(n_cols, device="cuda", dtype=torch.float32)
    bias = torch.randn(n_cols, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="超过单个 program 的支持范围"):
        layer_norm(x, weight, bias)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
