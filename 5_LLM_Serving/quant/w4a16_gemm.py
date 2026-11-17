"""
Marlin"""
import argparse

import torch
import triton
import triton.language as tl


M, N, K = 16, 64, 128
GROUP_K = 32

# 把 8 个 4-bit 数塞进 1 个 int32
def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    shifts = 4 * torch.arange(8, device=codes.device) # 生成位移量
    blocks = codes.reshape(K // 8, 8, N).to(torch.int64) # 沿 K 维，每 8 个切一组
    return (blocks << shifts[None, :, None]).sum(dim=1).to(torch.int32) # 左移并拼接：先广播位移向量然后与 blocks 对齐合并


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    shifts = 4 * torch.arange(8, device=packed.device) # 位移向量
    # 最高位为 1 时 packed 在 int32 中显示为负数。掩码只保留目标的 4 位。
    return ((packed[:, None, :] >> shifts[None, :, None]) & 0xF).reshape(K, N)


@triton.jit
def w4a16_kernel(
    a_ptr, packed_ptr, scale_ptr, out_ptr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SIZE_K: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 固定形状，只启动一个 program。A、打包权重和 scale 都是行主序。
    m = tl.arange(0, BLOCK_M)
    n = tl.arange(0, BLOCK_N)
    k_local = tl.arange(0, BLOCK_K)
    acc = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)

    for k_start in range(0, SIZE_K, BLOCK_K):
        k = k_start + k_local
        a = tl.load(a_ptr + m[:, None] * SIZE_K + k[None, :])

        # packed_B 是 [K // 8, N]。同一列的 8 个连续 k 共用一个打包字。
        packed_offsets = (k[:, None] // 8) * BLOCK_N + n[None, :]
        packed = tl.load(packed_ptr + packed_offsets)

        # 本实验 BLOCK_K 和量化组大小都是 32。
        # 当前 tile 只需为每个输出列加载一个 scale。
        group = k_start // BLOCK_K
        scale = tl.load(scale_ptr + group * BLOCK_N + n)

        # k % 8 是档位在打包字内的位置。右移后用掩码只取低 4 位。
        shift = 4 * (k[:, None] % 8)
        code = (packed >> shift) & 0xF

        # [32, 64] 的档位乘 [1, 64] 的 scale，得到当前 FP16 权重 tile。
        b = ((code.to(tl.float32) - 8) * scale[None, :]).to(tl.float16)
        acc = tl.dot(a, b, acc)

    tl.store(out_ptr + m[:, None] * BLOCK_N + n[None, :], acc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", action="store_true", help="运行补完后的 Triton kernel")
    args = parser.parse_args()

    rng = torch.Generator().manual_seed(0)
    a = (torch.randint(-8, 9, (M, K), generator=rng) / 8).half().cuda()
    codes = torch.randint(0, 16, (K, N), generator=rng)
    codes[:8, 0] = torch.arange(8, 16)
    packed = pack_int4(codes).cuda()
    scales = (torch.randint(1, 9, (K // GROUP_K, N), generator=rng) / 128).half().cuda()

    # PyTorch 参考先在显存中物化整块 FP16 权重，再做矩阵乘法。
    unpacked = unpack_int4(packed)
    torch.testing.assert_close(unpacked.cpu(), codes, rtol=0, atol=0)
    group = torch.arange(K, device=packed.device) // GROUP_K
    b_ref = ((unpacked.float() - 8) * scales[group].float()).half()
    y_ref = a @ b_ref

    first_word = int(packed[0, 0]) & 0xFFFFFFFF
    print(f"A={tuple(a.shape)}, packed_B={tuple(packed.shape)}, scales={tuple(scales.shape)}")
    print(f"packed_B[0,0]=0x{first_word:08x}, codes[:8,0]={unpacked[:8, 0].tolist()}")
    print(f"B[5,0]={b_ref[5, 0].item()}, scale[0,0]={scales[0, 0].item()}")
    print(f"pack/unpack round trip: OK, reference Y={tuple(y_ref.shape)}")

    if args.kernel:
        y = torch.empty_like(y_ref)
        w4a16_kernel[(1,)](a, packed, scales, y, M, N, K, GROUP_K, num_warps=4)
        torch.testing.assert_close(y, y_ref, rtol=1e-3, atol=1e-3)
        print(f"Triton vs PyTorch: OK, max_abs_error={(y - y_ref).abs().max().item():.6g}")


if __name__ == "__main__":
    main()
