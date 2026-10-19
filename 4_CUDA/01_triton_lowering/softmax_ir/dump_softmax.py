import inspect
import subprocess
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 一个 Triton Program 处理一整行
    row_idx = tl.program_id(axis=0)

    # 当前行的列索引
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 当前行在全局内存中的起始位置
    row_start = row_idx * n_cols

    # 当前行每个元素的地址
    input_ptrs = input_ptr + row_start + col_offsets

    # BLOCK_SIZE 可能大于真实列数
    mask = col_offsets < n_cols

    # 越界位置填 -inf，避免参与 max
    row = tl.load(
        input_ptrs,
        mask=mask,
        other=-float("inf"),
    )

    # 第一次 reduction：求整行最大值
    row_max = tl.max(row, axis=0)

    # Softmax 数值稳定化
    numerator = tl.exp(row - row_max)

    # 第二次 reduction：求指数和
    denominator = tl.sum(numerator, axis=0)

    # Softmax
    softmax_output = numerator / denominator

    # 写回当前行
    output_ptrs = output_ptr + row_start + col_offsets
    tl.store(
        output_ptrs,
        softmax_output,
        mask=mask,
    )


def dump_text(path: Path, content):
    path.write_text(str(content), encoding="utf-8")


def main():
    torch.manual_seed(0)

    # 1000 不是 2 的幂，因此 BLOCK_SIZE 会提升到 1024
    n_rows = 128
    n_cols = 1000
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # 故意使用多个 warp，后面重点观察 reduction lowering
    num_warps = 8

    x = torch.randn(
        (n_rows, n_cols),
        device="cuda",
        dtype=torch.float32,
    )
    y = torch.empty_like(x)

    grid = (n_rows,)

    # 先真正执行 kernel
    softmax_kernel[grid](
        x,
        y,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    # 与 PyTorch 做正确性比较
    y_ref = torch.softmax(x, dim=1)

    max_diff = (y - y_ref).abs().max().item()

    print(f"n_rows     = {n_rows}")
    print(f"n_cols     = {n_cols}")
    print(f"BLOCK_SIZE = {BLOCK_SIZE}")
    print(f"num_warps  = {num_warps}")
    print(f"max_diff   = {max_diff}")

    torch.testing.assert_close(
        y,
        y_ref,
        rtol=1e-4,
        atol=1e-5,
    )

    print("Correctness test: PASSED")

    # warmup 返回已经编译好的 kernel，便于读取各层 IR
    compiled = softmax_kernel.warmup(
        x,
        y,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        grid=grid,
    )

    ir_dir = Path(__file__).resolve().parent / "IR"
    ir_dir.mkdir(parents=True, exist_ok=True)

    # 保存 Python/Triton 源码
    source = inspect.getsource(softmax_kernel.fn)
    dump_text(ir_dir / "softmax_source.txt", source)

    print(f"Available asm keys: {compiled.asm.keys()}")

    # Triton 3.x 通常提供这些阶段
    if "ttir" in compiled.asm:
        dump_text(ir_dir / "softmax.ttir", compiled.asm["ttir"])

    if "ttgir" in compiled.asm:
        dump_text(ir_dir / "softmax.ttgir", compiled.asm["ttgir"])

    # Triton 当前接口中 LLVM IR 通常使用 llir 作为 key
    if "llir" in compiled.asm:
        dump_text(ir_dir / "softmax.ll", compiled.asm["llir"])

    if "ptx" in compiled.asm:
        dump_text(ir_dir / "softmax.ptx", compiled.asm["ptx"])
 
    if "cubin" in compiled.asm:
        cubin = compiled.asm["cubin"]

        if isinstance(cubin, bytes):
            (ir_dir / "softmax.cubin").write_bytes(cubin)
        else:
            # 某些版本可能返回类似 bytes 的对象
            (ir_dir / "softmax.cubin").write_bytes(bytes(cubin))

    # 使用 NVIDIA cuobjdump 反汇编真实机器码
    cubin_path = ir_dir / "softmax.cubin"
    sass_path = ir_dir / "softmax.sass"

    if cubin_path.exists():
        try:
            result = subprocess.run(
                ["cuobjdump", "--dump-sass", str(cubin_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            sass_path.write_text(result.stdout, encoding="utf-8")
        except FileNotFoundError:
            print("cuobjdump not found, skip SASS dump")
        except subprocess.CalledProcessError as e:
            print("cuobjdump failed:")
            print(e.stderr)

    print("\nDumped files:")
    for path in sorted(ir_dir.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()