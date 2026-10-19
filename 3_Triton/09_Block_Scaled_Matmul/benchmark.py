import argparse

import torch
import triton
import triton.language as tl
import triton.profiler as proton
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
from block_scaled_matmul import block_scaled_matmul


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == 'hip' and target.arch == 'gfx950'


def supports_block_scaling():
    return (is_cuda() and torch.cuda.get_device_capability()[0] in [10, 11]) or is_hip_cdna4()


def is_rubin():
    return torch.cuda.get_device_capability() == (10, 7)


if is_cuda() and torch.cuda.get_device_capability()[0] in [10, 11]:
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


def cublas_block_scaled_matmul(a, a_scale, b, b_scale, block_scale_type="mxfp8"):
    """
    cuBLAS block-scaled matmul baseline.

    Args:
        a: Input matrix A
            - For mxfp8: (M, K) in FP8 E4M3
            - For nvfp4: (M, K//2) in uint8 packed FP4 (2 elements per byte)
        a_scale: Scale factors for A
            - For mxfp8: E8M0 scales (flattened)
            - For nvfp4: FP8 E4M3 scales in cublas layout (M, K//16)
        b: Input matrix B
            - For mxfp8: (N, K) in FP8 E4M3
            - For nvfp4: (N, K//2) in uint8 packed FP4 (2 elements per byte)
        b_scale: Scale factors for B
            - For mxfp8: E8M0 scales (flattened)
            - For nvfp4: FP8 E4M3 scales in cublas layout (N, K//16)
        block_scale_type: Format type ("mxfp8" or "nvfp4")

    Returns:
        output: Result matrix (M, N) in FP16
    """
    M, K_a = a.shape
    N, K_b = b.shape

    if block_scale_type == "mxfp8":
        assert K_a == K_b, "K dimensions must match"
        assert a.dtype == torch.float8_e4m3fn, "Only FP8 E4M3 inputs supported for mxfp8"
        assert b.dtype == torch.float8_e4m3fn, "Only FP8 E4M3 inputs supported for mxfp8"
        # MXFP8 cuBLAS outputs FP16
        output = torch.empty((M, N), dtype=torch.float16, device="cuda")
        cublas.block_scaled_matmul_mxfp8(a, b, output, a_scale, b_scale)
    elif block_scale_type == "nvfp4":
        # For packed FP4, K_a and K_b are in bytes (K = K_a * 2 in elements)
        assert K_a == K_b, "K dimensions must match"
        assert a.dtype == torch.uint8, "Only uint8 packed FP4 inputs supported for nvfp4"
        assert b.dtype == torch.uint8, "Only uint8 packed FP4 inputs supported for nvfp4"
        # NVFP4 cuBLAS outputs FP16
        output = torch.empty((M, N), dtype=torch.float16, device="cuda")
        cublas.block_scaled_matmul_nvfp4(a, b, output, a_scale, b_scale)
    else:
        raise ValueError(f"Unsupported block_scale_type: {block_scale_type}")

    return output


def initialize_block_scaled(M, N, K, block_scale_type="nvfp4", compute_reference=False):
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256 if "fp4" in block_scale_type else 128
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    assert block_scale_type in ["nvfp4", "mxfp4", "mxfp8", "mixed"], f"Invalid block scale type: {block_scale_type}"
    ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1
    ELEM_PER_BYTE_B = 1 if block_scale_type == "mxfp8" else 2

    device = "cuda"
    a_ref = MXFP4Tensor(size=(M, K), device=device).random()
    # Similar to Hopper's wgmma symmetric fp8 instruction, the RHS is expected
    # to be in col-major layout for Blackwell's tcgen05.mma when using fp4 operands.
    # To conform to the expected semantics of tl.dot_scaled, (M, K) x (K, N),
    # the data is generated in col-major layout, packed along K for fp4, and then
    # logically transposed. Note that if one operand is of fp8 precision, unlike Hopper,
    # Blackwell supports both row-major and col-major layouts for the RHS matrix.
    # For the mixed-precision case, the fp4 RHS can be either in row or col-major layout.
    # But for performance reason, it is recommended to use col-major layout. If TMA is used
    # for the fp4 RHS operand load in mixed-precision dot, as in this tutorial, it must be
    # in col-major layout.
    b_ref = MXFP4Tensor(size=(N, K), device=device).random()
    if block_scale_type in ["mxfp8", "mixed"]:
        a_ref = a_ref.to(torch.float32)
        a = a_ref.to(torch.float8_e4m3fn)
    else:
        # Pack two fp4 elements per byte along K
        a = a_ref.to_packed_tensor(dim=1)

    if block_scale_type == "mxfp8":
        b_ref = b_ref.to(torch.float32)
        b = b_ref.to(torch.float8_e4m3fn)
    else:
        b = b_ref.to_packed_tensor(dim=1)

    b_ref = b_ref.to(torch.float32).T

    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K // ELEM_PER_BYTE_A])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K // ELEM_PER_BYTE_B])

    a_scale_shape = [M // 128, K // VEC_SIZE // 4, 32, 16]
    b_scale_shape = [N // 128, K // VEC_SIZE // 4, 32, 16]
    epsilon = 1e-8
    a_scale = torch.rand(a_scale_shape, device=device) + epsilon
    b_scale = torch.rand(b_scale_shape, device=device) + epsilon

    # Store original scales for cublas nvfp4 before any layout conversion.
    # For cublas nvfp4, the scales are in the original 4D layout.
    a_scale_orig = a_scale.clone()
    b_scale_orig = b_scale.clone()

    if block_scale_type == "nvfp4":
        a_scale = a_scale.to(torch.float8_e4m3fn)
        b_scale = b_scale.to(torch.float8_e4m3fn)
        a_scale_ref = a_scale
        b_scale_ref = b_scale
    elif block_scale_type in ["mxfp4", "mxfp8", "mixed"]:
        a_scale_ref = MXScaleTensor(a_scale)
        b_scale_ref = MXScaleTensor(b_scale)
        a_scale = a_scale_ref.data
        b_scale = b_scale_ref.data

    rep_m = BLOCK_M // 128
    rep_n = BLOCK_N // 128
    rep_k = BLOCK_K // VEC_SIZE // 4

    # Use 5D TMA descriptor [1, rep_m, rep_k, 2, 256] with uint8 elements.
    # With 256 elements we better utilize the L2 and don't require the TMA
    # engine to emit many small messages (16B) messages as with 32x16xu8.
    a_scale_block_shape = [1, rep_m, rep_k, 2, 256]
    b_scale_block_shape = [1, rep_n, rep_k, 2, 256]
    a_scale = a_scale.reshape(1, a_scale_shape[0], a_scale.shape[1], 2, 256)
    b_scale = b_scale.reshape(1, b_scale_shape[0], b_scale.shape[1], 2, 256)
    a_scale_desc = TensorDescriptor.from_tensor(a_scale, block_shape=a_scale_block_shape)
    b_scale_desc = TensorDescriptor.from_tensor(b_scale, block_shape=b_scale_block_shape)

    reference = None
    if compute_reference:
        a_scale_ref = a_scale_ref.to(torch.float32)
        b_scale_ref = b_scale_ref.to(torch.float32)

        def unpack_scale(packed):
            packed = packed.reshape(*packed.shape[:-2], 32, 4, 4)
            num_chunk_m, num_chunk_k, _, _, _ = packed.shape
            return packed.permute(0, 3, 2, 1, 4).reshape(num_chunk_m * 128, num_chunk_k * 4).contiguous()

        a_scale_ref = unpack_scale(a_scale_ref).repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
        b_scale_ref = unpack_scale(b_scale_ref).repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
        reference = torch.matmul(a_ref.to(torch.float32) * a_scale_ref, b_ref * b_scale_ref)

    if is_rubin():
        num_stages = 6
    else:
        num_stages = 4

    configs = {
        "BLOCK_SIZE_M": BLOCK_M,
        "BLOCK_SIZE_N": BLOCK_N,
        "BLOCK_SIZE_K": BLOCK_K,
        "num_stages": num_stages,
        "ELEM_PER_BYTE_A": ELEM_PER_BYTE_A,
        "ELEM_PER_BYTE_B": ELEM_PER_BYTE_B,
        "VEC_SIZE": VEC_SIZE,
        "disallow_acc_multi_buffer": not is_rubin(),
    }

    # Flatten scales for cuBLAS
    if block_scale_type == "mxfp8":
        a_scale_cublas = a_scale.contiguous().flatten()
        b_scale_cublas = b_scale.contiguous().flatten()
    elif block_scale_type == "nvfp4":
        a_scale_orig = a_scale_orig.to(torch.float8_e4m3fn)
        b_scale_orig = b_scale_orig.to(torch.float8_e4m3fn)
        a_scale_cublas = a_scale_orig.contiguous().flatten()
        b_scale_cublas = b_scale_orig.contiguous().flatten()
    else:
        a_scale_cublas = None
        b_scale_cublas = None

    return a_desc, a_scale_desc, b_desc, b_scale_desc, rep_m, rep_n, rep_k, configs, reference, a, b, a_scale_cublas, b_scale_cublas


def validate_block_scaled(M, N, K, block_scale_type="nvfp4", clc=False):
    results = initialize_block_scaled(M, N, K, block_scale_type, compute_reference=True)
    a_desc, a_scale_desc, b_desc, b_scale_desc, rep_m, rep_n, rep_k, configs, reference = results[:9]
    a, b, a_scale_cublas, b_scale_cublas = results[9:]

    # Test Triton implementation
    output = block_scaled_matmul(a_desc, a_scale_desc, b_desc, b_scale_desc, torch.float16, M, N, K, rep_m, rep_n,
                                 rep_k, configs, clc=clc)
    torch.testing.assert_close(reference, output.to(torch.float32), atol=1e-3, rtol=1e-3)

    # Test cuBLAS implementation if available (available for mxfp8 and nvfp4 only as of 13.1)
    if cublas and block_scale_type in ["mxfp8", "nvfp4"]:
        cublas_output = cublas_block_scaled_matmul(a, a_scale_cublas, b, b_scale_cublas,
                                                   block_scale_type=block_scale_type)
        torch.testing.assert_close(reference, cublas_output.to(torch.float32), atol=1e-3, rtol=1e-3)
        print(f"✅ (pass {block_scale_type} - Triton and cuBLAS)")
    else:
        print(f"✅ (pass {block_scale_type} - Triton only)")


def bench_block_scaled(K, block_scale_type="nvfp4", reps=10, warmup_reps=10, clc=False):
    assert K % 128 == 0
    M = 8192
    N = 8192
    print(f"Problem Shape = {M}x{N}x{K}")

    results = initialize_block_scaled(M, N, K, block_scale_type, compute_reference=False)
    a_desc, a_scale_desc, b_desc, b_scale_desc, rep_m, rep_n, rep_k, configs, _ = results[:9]
    a, b, a_scale_cublas, b_scale_cublas = results[9:]

    # Warmup
    for _ in range(warmup_reps):
        _ = block_scaled_matmul(a_desc, a_scale_desc, b_desc, b_scale_desc, torch.float16, M, N, K, rep_m, rep_n, rep_k,
                                configs, clc=clc)
        if cublas is not None and supports_block_scaling() and block_scale_type in ["mxfp8", "nvfp4"]:
            _ = cublas_block_scaled_matmul(a, a_scale_cublas, b, b_scale_cublas, block_scale_type=block_scale_type)

    # Benchmark
    proton.activate()
    for _ in range(reps):
        _ = block_scaled_matmul(a_desc, a_scale_desc, b_desc, b_scale_desc, torch.float16, M, N, K, rep_m, rep_n, rep_k,
                                configs, clc=clc)
        if cublas is not None and supports_block_scaling() and block_scale_type in ["mxfp8", "nvfp4"]:
            bytes_per_elem = a.element_size()
            # For nvfp4, K is in elements but a.shape[1] is in bytes, so use K/2 for byte calculation
            K_bytes = K if block_scale_type == "mxfp8" else K // 2
            with proton.scope(f"cublas [M={M}, N={N}, K={K}]",
                              {"bytes": bytes_per_elem * (M * K_bytes + N * K_bytes + M * N), "flops": 2. * M * N * K}):
                _ = cublas_block_scaled_matmul(a, a_scale_cublas, b, b_scale_cublas, block_scale_type=block_scale_type)
    proton.deactivate()
    print("Done benchmarking")


def show_profile(profile_name):
    import triton.profiler.viewer as proton_viewer

    metric_names = ["time/ms"]
    metric_names = ["tflop/s"] + metric_names
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)


@triton.jit
def block_scaled_matmul_kernel_cdna4(a_ptr, b_ptr, c_ptr, a_scales_ptr, b_scales_ptr, M, N, K, stride_am, stride_ak,
                                     stride_bk, stride_bn, stride_ck, stride_cm, stride_cn, stride_asm, stride_ask,
                                     stride_bsn, stride_bsk,
                                     # Meta-parameters
                                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                     mfma_nonkdim: tl.constexpr):
    """Kernel for computing the matmul C = A x B.
    A and B inputs are in the microscale fp4 (mxfp4) format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # We assume 32 elements along K share the same scale.
    SCALE_GROUP_SIZE: tl.constexpr = 32
    num_k_iter = tl.cdiv(K, BLOCK_K // 2)
    # Create pointers for first block of A and B input matrices
    # The BLOCK sizes are of the elements and in fp4 we pack 2 per uint8 container.
    offs_k = tl.arange(0, BLOCK_K // 2)
    offs_k_split = offs_k
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Create pointers for the first block of A and B scales
    offs_asn = (pid_n * (BLOCK_N // 32) + tl.arange(0, (BLOCK_N // 32))) % N
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP_SIZE * 32)

    # B scales are N x K even though B operand is K x N.
    b_scale_ptrs = (b_scales_ptr + offs_asn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk)
    offs_asm = (pid_m * (BLOCK_M // 32) + tl.arange(0, (BLOCK_M // 32))) % M
    a_scale_ptrs = (a_scales_ptr + offs_asm[:, None] * stride_asm + offs_ks[None, :] * stride_ask)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, num_k_iter):
        # Here we "undo" the shuffle done in global memory (shuffle_scales_cdna4 function).
        if mfma_nonkdim == 32:
            a_scales = tl.load(a_scale_ptrs).reshape(BLOCK_M // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 2, 32, 4,
                                                     1).permute(0, 3, 1, 4, 2,
                                                                5).reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)
            b_scales = tl.load(b_scale_ptrs).reshape(BLOCK_N // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 2, 32, 4,
                                                     1).permute(0, 3, 1, 4, 2,
                                                                5).reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)
        elif mfma_nonkdim == 16:
            a_scales = tl.load(a_scale_ptrs).reshape(BLOCK_M // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2,
                                                     1).permute(0, 5, 3, 1, 4, 2,
                                                                6).reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)
            b_scales = tl.load(b_scale_ptrs).reshape(BLOCK_N // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2,
                                                     1).permute(0, 5, 3, 1, 4, 2,
                                                                6).reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)

        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs, cache_modifier=None)

        accumulator += tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1")

        # Advance the ptrs to the next K block.
        a_ptrs += (BLOCK_K // 2) * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk

        a_scale_ptrs += BLOCK_K * stride_ask
        b_scale_ptrs += BLOCK_K * stride_bsk

    c = accumulator.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_ptrs = (c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :])
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")


def shuffle_scales_cdna4(scales: torch.Tensor, mfma_nonkdim: int):
    scales_shuffled = scales.clone()
    sm, sn = scales_shuffled.shape

    if mfma_nonkdim == 32:
        scales_shuffled = scales_shuffled.view(sm // 32, 32, sn // 8, 4, 2, 1)
        scales_shuffled = scales_shuffled.permute(0, 2, 4, 1, 3, 5).contiguous()
    elif mfma_nonkdim == 16:
        scales_shuffled = scales_shuffled.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
        scales_shuffled = scales_shuffled.permute(0, 3, 5, 2, 4, 1, 6).contiguous()

    scales_shuffled = scales_shuffled.view(sm // 32, sn * 32)
    return scales_shuffled


def initialize_block_scaled_amd(M, N, K, mfma_nonkdim):

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 256
    configs = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "num_stages": 2,
        "num_warps": 8,
        "mfma_nonkdim": mfma_nonkdim,
    }

    torch.manual_seed(5)

    x = MXFP4Tensor(size=(M, K), device="cuda").random()
    w = MXFP4Tensor(size=(N, K), device="cuda").random()

    x_scales = torch.randint(124, 128, (K // 32, M), dtype=torch.uint8, device="cuda")
    w_scales = torch.randint(124, 128, (K // 32, N), dtype=torch.uint8, device="cuda")
    x_scales = x_scales.T
    w_scales = w_scales.T
    x_scales_shuffled = shuffle_scales_cdna4(x_scales, configs["mfma_nonkdim"])
    w_scales_shuffled = shuffle_scales_cdna4(w_scales, configs["mfma_nonkdim"])

    return (
        x,
        w,
        x_scales,
        w_scales,
        x_scales_shuffled,
        w_scales_shuffled,
        configs,
    )


def validate_block_scaled_amd(M, N, K, block_scale_type="mxfp4", mfma_nonkdim=16):

    def e8m0_to_f32(x):
        x_f32 = 2**((x - 127).to(torch.float32))
        x_f32[x_f32 == 128] = float("nan")
        return x_f32

    def run_torch(x, w, x_scales, w_scales, dtype):
        # First convert the x and w inputs to f32.
        x_f32 = x.to(torch.float32)
        w_f32 = w.to(torch.float32)
        # Next convert the e8m0 scales to f32.
        x_scales = x_scales.repeat_interleave(32, dim=1).to(torch.float32)
        x_scales_f32 = e8m0_to_f32(x_scales)
        x_f32 = x_f32 * x_scales_f32
        w_scales = w_scales.repeat_interleave(32, dim=1).to(torch.float32)
        w_scales_f32 = e8m0_to_f32(w_scales)
        w_f32 = w_f32 * w_scales_f32
        return torch.mm(x_f32, w_f32.T).to(dtype)

    x_mxfp4, w_mxfp4, x_scales, w_scales, x_scales_triton, w_scales_triton, configs = \
    initialize_block_scaled_amd(M, N, K, mfma_nonkdim)

    x = x_mxfp4.to_packed_tensor(dim=1)
    w = w_mxfp4.to_packed_tensor(dim=1)

    triton_out = torch.empty((M, N), device=x.device)
    triton_out = block_scaled_matmul_amd(x, w, x_scales_triton, w_scales_triton, configs)
    triton_out = triton_out.to(torch.float32)

    torch_out = run_torch(x_mxfp4, w_mxfp4, x_scales, w_scales, torch.float32)
    torch.testing.assert_close(torch_out, triton_out)
    print(f"✅ (pass {block_scale_type}, mfma_nonk_dim {mfma_nonkdim})")


def block_scaled_matmul_amd(x, w, x_scales_triton, w_scales_triton, configs):
    M, K = x.shape
    N, K = w.shape
    w = w.T
    triton_out = torch.empty((M, N), device=x.device)

    kernel_kwargs = {}
    kernel_kwargs["matrix_instr_nonkdim"] = configs["mfma_nonkdim"]

    BLOCK_M = configs["BLOCK_M"]
    BLOCK_N = configs["BLOCK_N"]

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)

    triton_out = torch.empty((M, N), device="cuda")

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    block_scaled_matmul_kernel_cdna4[grid](x, w, triton_out, x_scales_triton, w_scales_triton, M, N, K, x.stride(0),
                                           x.stride(1), w.stride(0), w.stride(1), 0, triton_out.stride(0),
                                           triton_out.stride(1), x_scales_triton.stride(0), x_scales_triton.stride(1),
                                           w_scales_triton.stride(0), w_scales_triton.stride(1), BLOCK_M, BLOCK_N,
                                           configs["BLOCK_K"], configs["mfma_nonkdim"], num_warps=configs["num_warps"],
                                           num_stages=configs["num_stages"], **kernel_kwargs)
    triton_out = triton_out.to(torch.float32)

    return triton_out


def bench_block_scaled_amd(K, block_scale_type="mxfp4", reps=10, mfma_nonkdim=16):
    assert K % 128 == 0
    M = 8192
    N = 8192
    print(f"Problem Shape = {M}x{N}x{K}")

    x_mxfp4, w_mxfp4, x_scales, w_scales, x_scales_triton, w_scales_triton, configs = \
    initialize_block_scaled_amd(M, N, K, mfma_nonkdim)

    x = x_mxfp4.to_packed_tensor(dim=1)
    w = w_mxfp4.to_packed_tensor(dim=1)

    proton.activate()
    for _ in range(reps):
        _ = block_scaled_matmul_amd(x, w, x_scales_triton, w_scales_triton, configs)
    proton.deactivate()
    print("Done benchmarking")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, required=False, default=512)
    parser.add_argument("--K_range", type=int, nargs=2)
    parser.add_argument("--K_step", type=int, default=512)
    parser.add_argument("--bench", action="store_true", default=True)
    parser.add_argument("--format", type=str, choices=["mxfp4", "nvfp4", "mxfp8", "mixed"], default="nvfp4")
    parser.add_argument("--clc", action="store_true", help="Enable CLC scheduling on NVIDIA SM100+")
    args = parser.parse_args()

    if args.clc and (not is_cuda() or torch.cuda.get_device_capability()[0] < 10):
        parser.error("--clc requires an NVIDIA SM100+ GPU")

    if not supports_block_scaling():
        print("⛔ This example requires GPU support for block scaled matmul")
    else:
        if args.K and args.K_range is None:
            args.K_range = [args.K, args.K]
            args.K_step = 1  # doesn't matter as long as it's not 0

        torch.manual_seed(42)

        if is_cuda():
            validate_block_scaled(8192, 8192, 8192, block_scale_type=args.format, clc=args.clc)
        elif is_hip_cdna4():
            assert args.format == "mxfp4", "AMD tutorial only supports mxpf4 format currently"
            validate_block_scaled_amd(8192, 8192, 8192, block_scale_type=args.format, mfma_nonkdim=16)
            validate_block_scaled_amd(8192, 8192, 8192, block_scale_type=args.format, mfma_nonkdim=32)

        if args.bench:
            proton.start("block_scaled_matmul", hook="triton")
            proton.deactivate()  # Skip argument creation
            for K in range(args.K_range[0], args.K_range[1] + 1, args.K_step):
                if is_cuda():
                    bench_block_scaled(K, reps=10000, block_scale_type=args.format, clc=args.clc)
                elif is_hip_cdna4():
                    bench_block_scaled_amd(K, reps=10000, block_scale_type=args.format, mfma_nonkdim=16)
                    bench_block_scaled_amd(K, reps=10000, block_scale_type=args.format, mfma_nonkdim=32)
            proton.finalize()
            show_profile("block_scaled_matmul")