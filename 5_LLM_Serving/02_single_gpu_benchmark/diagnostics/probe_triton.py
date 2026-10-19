"""用小张量区分 PyTorch 基础执行、Triton JIT 和 Inductor 编译问题。"""

import argparse

import torch
import triton
import triton.language as tl
from triton.backends.nvidia.compiler import get_ptxas


@triton.jit
def add_one_kernel(source, destination, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(source + offsets, mask=offsets < SIZE, other=0)
    tl.store(destination + offsets, values + 1, mask=offsets < SIZE)


def add_one(source):
    return source + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["jit", "compile"], default="jit")
    args = parser.parse_args()
    capability = torch.cuda.get_device_capability(0)
    tool = get_ptxas(capability[0] * 10 + capability[1])
    print(f"Mode: {args.mode}", flush=True)
    print(f"Torch: {torch.__version__}; Triton: {triton.__version__}", flush=True)
    print(f"GPU capability: {capability}", flush=True)
    print(f"Selected ptxas: {tool.path}; CUDA tool version: {tool.version}", flush=True)

    source = torch.arange(256, device="cuda", dtype=torch.float32)
    expected = add_one(source)
    torch.cuda.synchronize()
    print("PYTORCH_EAGER_PASS", flush=True)

    if args.mode == "jit":
        actual = torch.empty_like(source)
        add_one_kernel[(1,)](source, actual, SIZE=256, BLOCK=256)
    else:
        compiled = torch.compile(add_one, backend="inductor", fullgraph=True)
        actual = compiled(source)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected)
    print(f"{args.mode.upper()}_PASS", flush=True)


if __name__ == "__main__":
    main()
