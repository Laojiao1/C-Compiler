import pytest
import torch

if __package__:
    from .kernel import add
else:
    # 兼容直接执行当前测试文件：python test_correctness.py
    from kernel import add


@pytest.mark.skipif(
        not torch.cuda.is_available(), reason="需要 CUDA GPU"
        )
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("size", [0, 1, 4096, 4096 + 123])
@pytest.mark.parametrize("block_size", [256, 512, 1024, 2048])
def test_add_correctness(size, dtype, block_size):
    """覆盖空输入、最小输入、对齐边界和 mask 尾块。"""
    torch.manual_seed(0)
    x = torch.rand(size, device="cuda", dtype=dtype)
    y = torch.rand(size, device="cuda", dtype=dtype)

    expected = x + y
    actual = add(x, y, BLOCK_SIZE=block_size)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")
def test_add_rejects_mismatched_dtype():
    x = torch.rand(128, device="cuda", dtype=torch.float32)
    y = torch.rand(128, device="cuda", dtype=torch.float16)

    with pytest.raises(TypeError, match="dtype"):
        add(x, y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")
def test_add_rejects_non_power_of_two_block_size():
    x = torch.rand(128, device="cuda", dtype=torch.float32)
    y = torch.rand(128, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="2 的幂"):
        add(x, y, BLOCK_SIZE=300)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
