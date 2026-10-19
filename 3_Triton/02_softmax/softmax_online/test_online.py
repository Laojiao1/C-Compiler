import pytest
import torch
import torch.nn.functional as F

if __package__:
    from .kernel_online import softmax_online
else:
    # 兼容直接执行当前测试文件：python test_online.py
    from kernel_online import softmax_online


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")
def test_online_correctness():
    """列数故意不对齐，确保最后一个 chunk 的 mask 参与计算。"""
    torch.manual_seed(42)
    n_rows = 4
    n_cols = 4096 + 123
    block_size = 256

    x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)

    # 把稍大的最大值放在靠后的 chunk。alpha 仍明显非零，因此旧贡献不能被直接丢弃。
    spike_col = n_cols - 17
    x[0, spike_col] = 8.0

    expected = F.softmax(x, dim=1)
    actual = softmax_online(x, BLOCK_SIZE=block_size)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(actual.sum(dim=1), torch.ones(n_rows, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")
def test_online_rejects_non_power_of_two_block_size():
    x = torch.randn(2, 128, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="2 的幂"):
        softmax_online(x, BLOCK_SIZE=300)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
