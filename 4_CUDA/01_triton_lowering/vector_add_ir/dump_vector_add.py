from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 当前 Triton Program 的编号
    pid = tl.program_id(axis=0)

    # 当前 Program 负责的逻辑元素下标
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 防止最后一个 Program 越界
    mask = offsets < n_elements

    # Global Memory -> Triton tensor
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # element-wise FP32 add
    output = x + y

    # Triton tensor -> Global Memory
    tl.store(output_ptr + offsets, output, mask=mask)

# 保存 ir 文件脚本
def save_ir(compiled_kernel):
    # 当前脚本目录
    base_dir = Path(__file__).parent

    # 所有 IR / binary 统一保存在 IR 目录
    ir_dir = base_dir / "IR"
    ir_dir.mkdir(exist_ok=True)

    print("ASM keys:", compiled_kernel.asm.keys())

    # 文本形式的中间产物
    text_outputs = {
        "source": "vector_add_source.txt",
        "ttir": "vector_add.ttir",
        "ttgir": "vector_add.ttgir",
        "llir": "vector_add.ll",
        "ptx": "vector_add.ptx",
    }

    for key, filename in text_outputs.items():
        if key in compiled_kernel.asm:
            content = compiled_kernel.asm[key]

            output_path = ir_dir / filename
            output_path.write_text(str(content), encoding="utf-8")

            print(f"Saved {key:6s}: {output_path}")

    # cubin 是二进制文件，不能 write_text
    if "cubin" in compiled_kernel.asm:
        cubin = compiled_kernel.asm["cubin"]

        output_path = ir_dir / "vector_add.cubin"

        if isinstance(cubin, bytes):
            output_path.write_bytes(cubin)
        else:
            # 某些版本 / backend 可能返回 bytearray 等
            output_path.write_bytes(bytes(cubin))

        print(f"Saved cubin : {output_path}")


def main():
    assert torch.cuda.is_available()

    print("Triton :", triton.__version__)
    print("PyTorch:", torch.__version__)
    print("GPU    :", torch.cuda.get_device_name())

    # 故意选一个不能被 256 整除的长度，
    # 后面观察 mask lowering 会更直观。
    n_elements = 1000
    BLOCK_SIZE = 256

    x = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    y = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    output = torch.empty_like(x)

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # 启动 kernel，同时获得编译后的 CompiledKernel
    compiled_kernel = vector_add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    # 确保 kernel 已执行完成
    torch.cuda.synchronize()

    # 先确认计算结果没有问题
    torch.testing.assert_close(output, x + y)
    print("Vector Add correctness: PASS")

    # 查看当前 Triton 实际暴露了哪些编译阶段
    # print("ASM keys:", compiled_kernel.asm.keys())

    # 取出真实 TTIR
    # ttir = compiled_kernel.asm["ttir"]

    # 保存 IR / binary 
    save_ir(compiled_kernel)


if __name__ == "__main__":
    main()