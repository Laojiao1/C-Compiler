import torch
import triton
import triton.language as tl
from triton.tools.disasm import get_sass
import os

@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 当前 program 负责一个 C tile
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # 当前 tile 对应的行列范围
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # K 维循环
    offs_k = tl.arange(0, BLOCK_K)

    # A tile 地址
    a_ptrs = A_ptr + (
        offs_m[:, None] * stride_am +
        offs_k[None, :] * stride_ak
    )

    # B tile 地址
    b_ptrs = B_ptr + (
        offs_k[:, None] * stride_bk +
        offs_n[None, :] * stride_bn
    )

    # 累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K 方向分块累加
    for k in range(0, K, BLOCK_K):
        k_offsets = k + offs_k

        a = tl.load(
            a_ptrs + k * stride_ak, # 动态计算当前步的绝对地址
            mask=(
                (offs_m[:, None] < M) &
                (k_offsets[None, :] < K)
            ),
            other=0.0
        )

        b = tl.load(
            b_ptrs + k * stride_bk,
            mask=(
                (k_offsets[:, None] < K) &
                (offs_n[None, :] < N)
            ),
            other=0.0
        )

        # 核心：Triton dot 会在 lowering 阶段展开到 GPU 矩阵乘路径
        acc += tl.dot(a, b)

    # fp32 accumulator 转 fp16
    c = acc.to(tl.float16)

    c_ptrs = C_ptr + (
        offs_m[:, None] * stride_cm +
        offs_n[None, :] * stride_cn
    )

    mask = (
        (offs_m[:, None] < M) &
        (offs_n[None, :] < N)
    )

    tl.store(c_ptrs, c, mask=mask)


def matmul(
    a,
    b
):
    M, K = a.shape
    K, N = b.shape

    c = torch.empty(
        (M, N),
        device=a.device,
        dtype=torch.float16
    )

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
    )

    return c


def test():
    torch.manual_seed(0)

    M = 128
    N = 128
    K = 128

    A = torch.randn(
        (M, K),
        device="cuda",
        dtype=torch.float16
    )

    B = torch.randn(
        (K, N),
        device="cuda",
        dtype=torch.float16
    )

    C = matmul(A, B)

    ref = torch.matmul(A, B)

    max_error = torch.max(
        torch.abs(
            C.float() - ref.float()
        )
    ).item()

    print("max error:", max_error)

    torch.testing.assert_close(
        C,
        ref,
        atol=2e-2,
        rtol=2e-2,
    )

    print("matmul test passed")


def main():
    compiled = matmul_kernel.warmup(
        torch.empty((128,128), device="cuda", dtype=torch.float16),
        torch.empty((128,128), device="cuda", dtype=torch.float16),
        torch.empty((128,128), device="cuda", dtype=torch.float16),
        128,
        128,
        128,
        128,
        1,
        128,
        1,
        128,
        1,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        grid=(2,2),
    )

    # 编译 asm
    asm = compiled.asm

    # 手动将反汇编生成的 SASS 填充进 asm 字典中
    if "cubin" in asm:
        try:
            asm["sass"] = get_sass(asm["cubin"])
            print("Successfully disassembled CUBIN to SASS!")
        except Exception as e:
            print(f"Failed to generate SASS via get_sass: {e}")

    # 获取当前 .py 脚本所在的绝对路径目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # 拼接出同级 IR 目录路径
    ir_dir = os.path.join(current_script_dir, "IR")

    # 自动创建 IR 目录（如果不存在的话）
    os.makedirs(ir_dir, exist_ok=True)

    # 安全写出文件
    for key, value in asm.items():
        if value is not None:
            file_path = os.path.join(ir_dir, f"matmul.{key}")

            if isinstance(value, bytes):
                with open(file_path, "wb") as f:
                    f.write(value)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(str(value))

    print(f"IR files dumped successfully into: {ir_dir}")


if __name__ == "__main__":
    test()
    main()
