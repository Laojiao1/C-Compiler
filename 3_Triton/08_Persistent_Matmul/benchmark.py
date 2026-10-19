import argparse
import itertools

import torch

import triton.language as tl
import triton.profiler as proton

from contextlib import contextmanager

import config

from config import (
    matmul,
    matmul_persistent,
    matmul_tma,
    matmul_tma_clc,
    matmul_tma_persistent,
    matmul_descriptor_persistent,
    HAS_TENSOR_DESC,
    HAS_HOST_TENSOR_DESC,
    HAS_TMA_CLC,
    HAS_WARP_SPECIALIZE,
    is_hopper
)

def device_blas_matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    blas_name = config.device_blas_name()
    with proton.scope(f"{blas_name} [M={M}, N={N}, K={K}]",
                      {"bytes": bytes_per_elem * (M * K + N * K + M * N), flops_str: 2. * M * N * K}):
        config.device_blas.matmul(a, b, c)
    return c


def torch_matmul(a, b):
    M, K = a.shape
    N, K = b.shape
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    with proton.scope(f"torch [M={M}, N={N}, K={K}]",
                      {"bytes": bytes_per_elem * (M * K + N * K + M * N), flops_str: 2. * M * N * K}):
        c = torch.matmul(a, b.T)
    return c


@contextmanager
def proton_context():
    proton.activate(0)
    try:
        yield
    finally:
        proton.deactivate(0)


def bench_fn(label, reps, warmup_reps, fn, *args):
    print(f"Benchmarking {label}: ...", end="")
    for _ in range(warmup_reps):
        fn(*args)
    with proton_context():
        for _ in range(reps):
            fn(*args)
    print(f"\rBenchmarking {label}: done")


def bench(K, dtype, reps=10000, warmup_reps=10000):
    M = 8192
    N = 8192
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)

    b = b.T.contiguous()

    if config.device_blas is not None:
        blas_name = config.device_blas_name()
        bench_fn(blas_name, reps, warmup_reps, device_blas_matmul, a, b)
    if dtype == torch.float16:
        bench_fn("torch", reps, warmup_reps, torch_matmul, a, b)
    bench_fn("naive", reps, warmup_reps, matmul, a, b.T)
    bench_fn("persistent", reps, warmup_reps, matmul_persistent, a, b.T)
    warp_specialize = [False, True] if HAS_WARP_SPECIALIZE else [False]
    for ws in warp_specialize:
        ws_str = "_ws" if ws else ""
        # disable on-host warpspec on Hopper
        if HAS_HOST_TENSOR_DESC and not (is_hopper() and ws):
            if HAS_TMA_CLC:
                bench_fn(f"clc_tma{ws_str}", reps, warmup_reps, lambda a, b: matmul_tma_clc(a, b, ws), a, b)
            bench_fn(f"tma_persistent{ws_str}", reps, warmup_reps, lambda a, b: matmul_tma_persistent(a, b, ws), a, b)
            bench_fn(f"tma{ws_str}", reps, warmup_reps, lambda a, b: matmul_tma(a, b, ws), a, b)
        if HAS_TENSOR_DESC:
            bench_fn(f"descriptor_persistent{ws_str}", reps, warmup_reps,
                     lambda a, b: matmul_descriptor_persistent(a, b, ws), a, b)


def run_test(expect, fn, a, b, label, enabled=True):
    print(f"  {label}: ...", end="")
    if enabled:
        actual = fn(a, b)
        passed = torch.allclose(expect, actual.to(expect.dtype), atol=1.0)
        icon = "✅" if passed else "❌"
    else:
        icon = "⭕"
    print(f"\r  {label}: {icon}  ")


def validate(M, N, K, dtype):
    print(f"{M=}, {N=}, {K=}, verification naive vs: ")
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    b = b.T.contiguous()

    naive_result = matmul(a, b.T).to(torch.float16)
    run_test(naive_result, torch_matmul, a, b, "Torch", enabled=dtype == torch.float16)
    run_test(naive_result, device_blas_matmul, a, b, config.device_blas_name(), enabled=config.device_blas is not None)
    run_test(naive_result, matmul_persistent, a, b.T, "Persistent")
    kernels = [
        (matmul_tma, "TMA", HAS_HOST_TENSOR_DESC),
        (matmul_tma_clc, "CLC TMA", HAS_TMA_CLC),
        (matmul_tma_persistent, "TMA Persistent", HAS_HOST_TENSOR_DESC),
        (matmul_descriptor_persistent, "Tensor Descriptor Persistent", HAS_TENSOR_DESC),
    ]
    warp_specialize = [False, True] if HAS_WARP_SPECIALIZE else [False]

    for (kernel, label, enabled), warp_specialize in itertools.product(kernels, warp_specialize):
        label = f"{label} (warp_specialize={warp_specialize})"
        # skip if hopper and warp_specialize and not on-device
        skipped = is_hopper() and warp_specialize and kernel != matmul_descriptor_persistent
        enabled = enabled and (not warp_specialize or HAS_TENSOR_DESC) and (not skipped)
        run_test(naive_result, lambda a, b: kernel(a, b, warp_specialize), a, b, label, enabled)
    print()


def show_profile(precision, profile_name):
    import triton.profiler.viewer as proton_viewer
    metric_names = ["time/ms"]
    if precision == 'fp8':
        metric_names = ["tflop8/s"] + metric_names
    elif precision == 'fp16':
        metric_names = ["tflop16/s"] + metric_names
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, required=False, default=512)
    parser.add_argument("--K_range", type=int, nargs=2)
    parser.add_argument("--K_step", type=int, default=512)
    parser.add_argument("--prec", type=str, choices=["fp8", "fp16"], default="fp16")
    args = parser.parse_args()

    if args.prec == 'fp8' and (not hasattr(torch, "float8_e4m3fn") or not config.is_cuda()):
        print("This example requires CUDA/HIP with fp8 support.")
    else:
        dtype = torch.float8_e4m3fn if args.prec == 'fp8' else torch.float16

        if args.K and args.K_range is None:
            args.K_range = [args.K, args.K]
            args.K_step = 1  # doesn't matter as long as it's not 0

        torch.manual_seed(0)

        validate(32, 32, 32, dtype)
        validate(8192, 8192, args.K_range[0], dtype)

        proton.start("matmul", hook="triton")
        proton.deactivate()
        for K in range(args.K_range[0], args.K_range[1] + 1, args.K_step):
            bench(K, dtype)
        proton.finalize()
        show_profile(args.prec, "matmul")