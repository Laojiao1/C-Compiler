import pytest
import torch
import torch.nn.functional as F

if __package__:
    from .kernel import softmax
else:
    # 兼容直接执行当前测试文件：python test_correctness.py
    from kernel import softmax


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 1), (7, 127), (32, 1024), (16, 1024 + 123)])
def test_softmax_correctness(shape, dtype):
    """覆盖最小行、非 2 次幂 reduction、对齐和 mask 尾部。"""
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=dtype)

    # 极大值会让不减 row_max 的实现溢出，同时验证数值稳定路径。
    x[0, -1] = 1000.0

    expected = F.softmax(x, dim=1)
    actual = softmax(x)

    if dtype == torch.float16:
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    else:
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")
def test_softmax_empty_matrix():
    x = torch.empty((0, 128), device="cuda", dtype=torch.float32)
    actual = softmax(x)
    assert actual.shape == x.shape
    assert actual.numel() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")
def test_softmax_rejects_noncontiguous_last_dimension():
    x = torch.randn(17, 31, device="cuda", dtype=torch.float32).T
    assert x.stride(1) != 1

    with pytest.raises(ValueError, match="最后一维必须连续"):
        softmax(x)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
