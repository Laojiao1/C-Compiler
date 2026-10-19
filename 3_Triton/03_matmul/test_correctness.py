import pytest
import torch

if __package__:
    from .kernel import matmul, matmul_autotuned
else:
    # 兼容直接执行：python test_correctness.py
    from kernel import matmul, matmul_autotuned


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="需要 CUDA GPU",
)
@pytest.mark.parametrize(
    # 测试数据的形状
    "shape",
    [
        (32, 32, 32),  # M/N/K 都与 block 对齐
        (64, 48, 96),  # 非方阵，N 存在边缘 tile
        (37, 53, 29),  # M/N/K 全部需要 mask
        (1, 1, 1),     # 极小矩阵
    ],
)
def test_matmul_correctness(shape):
    M, N, K = shape

    torch.manual_seed(0)
    a = torch.randn(
        (M, K),
        device="cuda",
        dtype=torch.float16,
    )
    b = torch.randn(
        (K, N),
        device="cuda",
        dtype=torch.float16,
    )

    expected = torch.matmul(a, b)
    actual = matmul(a, b)

    torch.testing.assert_close(
        actual,
        expected,
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="需要 CUDA GPU",
)
def test_autotuned_matmul_correctness():
    M, N, K = 257, 263, 129

    torch.manual_seed(0)
    a = torch.randn(
        (M, K),
        device="cuda",
        dtype=torch.float16,
    )
    b = torch.randn(
        (K, N),
        device="cuda",
        dtype=torch.float16,
    )

    expected = torch.matmul(a, b)
    actual = matmul_autotuned(a, b)

    torch.testing.assert_close(
        actual,
        expected,
        atol=2e-2,
        rtol=2e-2,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
