import pytest
import torch
import torch.nn.functional as F

if __package__:
    from .kernel_FlashAtten import flash_attn_1d
else:
    # 兼容直接执行当前测试文件：python test_FlashAtten.py
    from kernel_FlashAtten import flash_attn_1d


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")
@pytest.mark.parametrize("d_v", [63, 64, 65])
def test_flash_attn_1d_correctness(d_v):
    """63/65 专门验证 D_V 向上补齐后不会越界读写。"""
    torch.manual_seed(123)
    n_rows = 4
    n_cols = 2048 + 17
    block_size = 256

    x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
    v = torch.randn(n_cols, d_v, device="cuda", dtype=torch.float32)

    # 使用适度增大的后段最大值，让旧 acc 以非零 alpha 缩放，真正检验重标定公式。
    x[0, n_cols - 9] = 8.0

    expected = F.softmax(x, dim=-1) @ v
    actual = flash_attn_1d(x, v, BLOCK_SIZE=block_size)

    assert actual.shape == (n_rows, d_v)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")
def test_flash_attn_rejects_incompatible_shapes():
    x = torch.randn(2, 128, device="cuda", dtype=torch.float32)
    v = torch.randn(127, 64, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match=r"v\.shape\[0\]"):
        flash_attn_1d(x, v)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
