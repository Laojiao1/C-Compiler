import torch
import triton
import triton.language as tl

# 测试不同的参数配置
AUTOTUNE_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 1},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 2},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 2},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 2},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 2},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 2},
        num_warps=8,
        num_stages=3,
    ),
]

# 参数定义
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
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
    GROUP_SIZE_M: tl.constexpr,
):
    # 将一维 program ID 按 M 方向分组，目的是提高相邻 program 的 L2Cache 数据复用
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # ----- 关键代码 -----
    # 以下这段代码把多个 program 揉成一个“组”（大小为 GROUP_SIZE_M）：为了优化遍历次序，提高 L2Cache 的数据复用
    # 本来水平一字排开的计算顺序，变成了在一个纵向宽度为 GROUP_SIZE_M 的长方形区域内“拐弯”扫描
    num_pid_in_group = GROUP_SIZE_M * num_pid_n # 计算一个组里有多少个 pid

    group_id = pid // num_pid_in_group # 计算当前program属于哪一组
    first_pid_m = group_id * GROUP_SIZE_M # 计算当前组从哪个M开始
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M) # 计算当前组实际有多少个M tile
    pid_in_group = pid % num_pid_in_group # 当前pid在组内的编号

    pid_m = first_pid_m + (pid_in_group % group_size_m) # M的方向tile
    pid_n = pid_in_group // group_size_m # N的方向tile，每经过group_size_m，才移动到下一个 n tile

    # C 的行来自 A，列来自 B
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # offs_k 描述当前 K tile 内部的位置
    offs_k = tl.arange(0, BLOCK_K)

    # A tile 的形状是 [BLOCK_M, BLOCK_K]
    # 我们需要将 A tile 的行索引映射到 A 的行索引，列索引映射到 A 的列索引
    a_ptrs = (
        a_ptr
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )

    # B tile 的形状是 [BLOCK_K, BLOCK_N]
    # 我们需要将 B tile 的行索引映射到 B 的行索引，列索引映射到 B 的列索引
    b_ptrs = (
        b_ptr
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    # 初始化累加结果：tl.dot 的输出形状是 [BLOCK_M, BLOCK_N]
    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )

    # 滑动窗口：在 K 维度上 每次处理 BLOCK_K 个 K 元素
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 最后一个 K tile 可能越界，越界元素按 0 处理。
        k_remaining = K - k * BLOCK_K

        a_mask = (
            (offs_m[:, None] < M)
            & (offs_k[None, :] < k_remaining)
        )

        b_mask = (
            (offs_k[:, None] < k_remaining)
            & (offs_n[None, :] < N)
        )

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        # → [BLOCK_M, BLOCK_N]
        # dot 在底层通常会映射到 Tensor Core 的硬件指令（如 MMA），执行高效的矩阵乘累加
        accumulator = tl.dot(a, b, accumulator)

        # 循环结束，加上步长，进入下一个 K tile：A 向右，B 向下
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 将结果写回 C
    c_ptrs = (
        c_ptr
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )

    # 由于 C 的形状可能不是 BLOCK_M x BLOCK_N，最后一个 tile 可能越界。
    c_mask = (
        (offs_m[:, None] < M)
        & (offs_n[None, :] < N)
    )

    tl.store(c_ptrs, accumulator, mask=c_mask)

# Python 包装函数
def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    BLOCK_M: int = 32,
    BLOCK_N: int = 32,
    BLOCK_K: int = 32,
    num_warps: int = 4,
    GROUP_SIZE_M: int = 8,
) -> torch.Tensor:
    # 条件检查
    assert a.ndim == 2
    assert b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert a.device == b.device
    assert a.dtype == b.dtype
    assert a.is_cuda and b.is_cuda

    M, K = a.shape
    _, N = b.shape

    # 输出结果c
    c = torch.empty(
        (M, N),
        device=a.device,
        dtype=a.dtype,
    )

    # 一维 grid；kernel 内部将 pid 映射为分组后的 (pid_m, pid_n)。
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

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
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
    )

    return c

# 自动调优版本的 kernel
autotuned_matmul_kernel = triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["M", "N", "K"],
)(matmul_kernel)

# Python 包装函数：自动调优版本
def matmul_autotuned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.ndim == 2
    assert b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert a.device == b.device
    assert a.dtype == b.dtype
    assert a.is_cuda and b.is_cuda

    M, K = a.shape
    _, N = b.shape

    c = torch.empty(
        (M, N),
        device=a.device,
        dtype=a.dtype,
    )

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"])
        * triton.cdiv(N, meta["BLOCK_N"]),
    )

    autotuned_matmul_kernel[grid](
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
    )

    return c

    
