import torch
import triton
import triton.language as tl

from triton.tools.tensor_descriptor import TensorDescriptor
from contextlib import contextmanager

import config

"""
采用 CLC 硬件级调度器：
    它直接在硬件层面上为整个 Cluster 统一分派下一个 2D 宏块（Macro Tile），并自动实现集群内的数据广播与同步
    编译器在生成 PTX/SASS 汇编时，会插入特定于硬件集群调度的元数据和指令，将原本由软件调度的单个 Block 绑定为硬件级 Threadblock Cluster
好处：
    将多 SM 协同的 L2 缓存复用和 Shared Memory 广播推进到了硬件流水线级别，是大模型超高吞吐推理与训练的终极优化手段
"""
@triton.autotune(
    configs=config.matmul_tma_persistent_get_configs(
        # 传入 CLC 依赖集群拓扑配置，例如 Cluster 的三维尺寸 cluster_dim = (2, 1, 1) 等
        pre_hook = config.matmul_tma_set_block_size_hook 
    ),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
)
@triton.jit(launch_metadata=config._matmul_launch_metadata)
def matmul_kernel_tma_clc(
    a_desc, b_desc, c_desc,  # TMA
    M, N, K,  
    BLOCK_SIZE_M: tl.constexpr,  
    BLOCK_SIZE_N: tl.constexpr,  
    BLOCK_SIZE_K: tl.constexpr,  
    GROUP_SIZE_M: tl.constexpr,  
    FP8_OUTPUT: tl.constexpr,  
    EPILOGUE_SUBTILE: tl.constexpr,  
    WARP_SPECIALIZE: tl.constexpr,  
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16

    tile_id = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n # 一个 Group 包含的 Tile 总数
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    group_id = tile_id // num_pid_in_group # 当前 Tile 属于第几个行分组
    first_pid_m = group_id * GROUP_SIZE_M # 该分组在 M 轴的起始基准
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M) # 处理 M 边缘不对齐

    pid_m = first_pid_m + (tile_id % group_size_m) # 沿 M 轴排列
    pid_n = (tile_id % num_pid_in_group) // group_size_m # 沿 N 轴推进

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


    """
    由于在外部传入了 clc=True，
    Triton 编译器与硬件 CLC 单元在底层接管了 Cluster 的任务分派与调度流水线，
    无需像 Persistent 版本那样写 NUM_SMS 步长循环
    """
    # warp_specialize 作用在内层 K 维度循环
    for ki in tl.range(k_tiles, warp_specialize=WARP_SPECIALIZE):
        offs_k = ki * BLOCK_SIZE_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

    if EPILOGUE_SUBTILE:
        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        c0 = acc0.to(dtype)
        c_desc.store([offs_am, offs_bn], c0)
        c1 = acc1.to(dtype)
        c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
    else:
        accumulator = accumulator.to(dtype)
        c_desc.store([offs_am, offs_bn], accumulator)


def matmul_tma_clc(a, b, warp_specialize: bool):
    assert config.HAS_TMA_CLC, "CLC TMA requires an NVIDIA SM100+ GPU with tensor descriptor support"
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    dummy_block = [1, 1]
    a_desc = TensorDescriptor.from_tensor(a, dummy_block)
    b_desc = TensorDescriptor.from_tensor(b, dummy_block)
    c_desc = TensorDescriptor.from_tensor(c, dummy_block)

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )

    matmul_kernel_tma_clc[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        WARP_SPECIALIZE=warp_specialize,  #
        clc=True,  # 开启 Triton 编译器的 CLC 模式，编译器与硬件 CLC 单元在底层接管了 Cluster 的任务分派与调度流水线
    )
    return c