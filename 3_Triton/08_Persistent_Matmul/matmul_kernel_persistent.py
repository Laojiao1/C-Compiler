import torch
import triton
import triton.language as tl

from triton.tools.tensor_descriptor import TensorDescriptor

from typing import Optional
import config

"""按列处理，提高 L2 cache 复用，降低主存带宽压力"""
@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


"""
Persistent 思想：
    只启动与 GPU 物理 SM 计算单元数量匹配的 Program（常驻线程块），
    每个 SM 永不退出，而是在一个大循环中主动按步长去任务池种拉取 Tile，
    从而彻底消除海量线程块的创建/注销的硬件开销，
    同时消除最后一个 wave 无法填满 SM 而造成的计算机资源闲置（简称 wave quantization 气泡）
"""
@triton.autotune(
    configs=config.matmul_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=config._matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr, b_ptr, c_ptr,  
    M, N, K,  
    stride_am, stride_ak,  
    stride_bk, stride_bn,  
    stride_cm, stride_cn,  
    BLOCK_SIZE_M: tl.constexpr,  
    BLOCK_SIZE_N: tl.constexpr,  
    BLOCK_SIZE_K: tl.constexpr,  
    GROUP_SIZE_M: tl.constexpr,  
    NUM_SMS: tl.constexpr,  
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # 初始化：提前倒扣 NUM_SMS，预留“边算下一块、边存上一块”的流水线
    # 目的：算第 i 块的同时，让硬件后台去存第 i-1 块
    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)

        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N

        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)

        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)

        # 向 Triton 编译器提供显式优化提示：强制编译器生成最高效的 128 bit 向量化访存指令
        # 内层 multiple_of：告诉编译器 offs_am 的基地址一定是 BLOCK_SIZE_M 的整数倍
        # 外层 max_contiguous：告诉编译器 假定这个张量在长度为 BLOCK_SIZE_M 的维度上，元素是严格连续递增（stride=1）的
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)

            accumulator = tl.dot(a, b, accumulator)

        # 流水线重叠：此时 tile_id_c 刚好等于当前算的那块
        tile_id_c += NUM_SMS

        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        if (c_ptr.dtype.element_ty == tl.float8e4nv):
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)

        tl.store(c_ptrs, c, mask=c_mask)


def matmul_persistent(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_persistent[grid](
        a, b, c,  
        M, N, K,  
        a.stride(0), a.stride(1),  
        b.stride(0), b.stride(1),  
        c.stride(0), c.stride(1),  
        NUM_SMS=NUM_SMS,  
    )
    return c


"""
采用：
    TMA 张量内存加速器 +
    Persistent Grid 持久化线程 +
    warp Specialization 线程束专业化 + 
    Epilogue Sub-tiling 尾声子块切分
"""
@triton.autotune(
    configs=config.matmul_tma_persistent_get_configs(pre_hook=config.matmul_tma_set_block_size_hook),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
)
@triton.jit(launch_metadata=config._matmul_launch_metadata)
def matmul_kernel_tma_persistent(
    # TMA 张量描述符：
    # 在 Host 端就打包好多维矩阵的全局基地址、形状、步长、切片尺寸（Box Size）以及边界填充模式
    a_desc, b_desc, c_desc,  
    M, N, K,  
    BLOCK_SIZE_M: tl.constexpr,  
    BLOCK_SIZE_N: tl.constexpr,  
    BLOCK_SIZE_K: tl.constexpr,  
    GROUP_SIZE_M: tl.constexpr,  
    FP8_OUTPUT: tl.constexpr,  
    EPILOGUE_SUBTILE: tl.constexpr,  
    NUM_SMS: tl.constexpr,  
    WARP_SPECIALIZE: tl.constexpr,  
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # 初始化预留流水线
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # 采用持久化线程思想：每次跳跃 NUM_SMS 个步长，直到处理完所有的 num_tiles
    # 采用线程束专业化思想：Triton 会在底层把同一个 Block 内的 Warp 拆分为搬运和计算
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)

        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            # warp 异步搬运：指挥 TMA 硬件单元，把数据从显存搬到共享内存（SRAM）。发完搬运指令就走，从不做矩阵乘计算
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])

            # warp 计算：守在 Tensor Core 旁边，只要 SRAM 里有数据就立刻开算并累加，算完负责尾声切分和写回
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS

        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)

        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        # ----- Epilogue Sub-tiling 尾声子块切分 -----
        """
            由于 acc 里存放的时 FP32，而最终输出需要转换为 PF16/FP8
            采用尾声子图优化：一种将计算和存储分解成多个部分的技巧，可以节省片上共享内存（SRAM）
            核心思想：算的时候整块算（吞吐高），存的时候切碎存（省资源）
        """
        if EPILOGUE_SUBTILE:
            # 这一段通过张量变换，把一个大矩阵在 N 维度上一分为二

            # 1. 重构形状：把 N 维拆成 2 份，每份 128 
            # (128, 2, 128)
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))

            # 2. 轴置换：把第 1 维（大小为 2 的那个维度）挪到最后面
            # (128, 128, 2) -> 本质上是两个 (128, 128) 的小块
            acc = tl.permute(acc, (0, 2, 1))

            # 3. 拆分张量：取出左半块 acc0 和右半块 acc1
            acc0, acc1 = tl.split(acc)

            # 拆完之后，程序按照：转一半 -> 存一半 -> 释放 -> 转另一半 -> 存另一半 执行
            # --- 处理左半块 (N 坐标从 offs_bn_c 开始) ---
            c0 = acc0.to(dtype) # 只对左半块做类型转换（只占一半额外寄存器）
            c_desc.store([offs_am_c, offs_bn_c], c0) # TMA 异步写回左半块，写完后 c0 立即释放

            # --- 处理右半块 (N 坐标偏移 BLOCK_SIZE_N // 2) ---
            c1 = acc1.to(dtype) # 对右半块做类型转换（复用刚刚释放的寄存器）
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        else: # 不用子块切分：为整块数据申请转换缓冲区
            """
                如果 Tile 很大，寄存器很容易被撑爆，发生 spill；
                通过 TMA 写回全局显存时，硬件通常需要在片上共享内存（SRAM）中开辟中转空间。一次性写回大块，需要占用翻倍的 SRAM 空间
            """
            accumulator = accumulator.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], accumulator)


def matmul_tma_persistent(a, b, warp_specialize: bool):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor.from_tensor(a, dummy_block)
    b_desc = TensorDescriptor.from_tensor(b, dummy_block)
    c_desc = TensorDescriptor.from_tensor(c, dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        ), )

    matmul_kernel_tma_persistent[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=warp_specialize,  #
    )
    return c


"""
    引入设备端张量描述符（Device-Side Tensor Descriptor）
    添加 FLATTEN 循环展开优化
"""
@triton.autotune(
        configs = config.matmul_tma_persistent_get_configs(), 
        key = ["M", "N", "K", "WARP_SPECIALIZE", "FLATTEN"],
        prune_configs_by = {'early_config_prune': config.prune_invalid_configs}
)
@triton.jit(launch_metadata = config._matmul_launch_metadata)
def matmul_kernel_descriptor_persistent(
    a_ptr, b_ptr, c_ptr,  
    M, N, K,  
    BLOCK_SIZE_M: tl.constexpr,  
    BLOCK_SIZE_N: tl.constexpr,  
    BLOCK_SIZE_K: tl.constexpr,  
    GROUP_SIZE_M: tl.constexpr,  
    EPILOGUE_SUBTILE: tl.constexpr,  
    NUM_SMS: tl.constexpr,  
    WARP_SPECIALIZE: tl.constexpr,  
    FLATTEN: tl.constexpr, 
    # 当 FLATTEN=True 时，Triton 编译器会将 Persistent 大循环与内层的 K 维循环在 LLVM IR 层面做拍平展开与调度融合
    # 好处：消除循环分支跳转带来的指令气泡，让编译器能够跨越外层 Tile 循环与内层 K 循环边界，更大范围地排布内存预取与 MMA 计算指令
):
    # 使用 TMA 和设备端描述符创建的 Matmul
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # 在 GPU 端动态创建 TMA 描述符
    # 好处：消除 Host 端开销与接口侵入，支持动态 Shape
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    # 尾声切分（Sub-tiling）与描述符块大小的动态适配
    # 确保后续调用 c_desc.store() 写入 c0 和 c1 时，硬件 TMA 搬运单元每次搬运的恰好是半个 Tile 的真实数据量，实现了描述符配置与下游张量拆分的严格对齐
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2],
    )

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=FLATTEN, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_cm, offs_cn], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], c1)
        else:
            c = accumulator.to(dtype)
            c_desc.store([offs_cm, offs_cn], c)


def matmul_descriptor_persistent(a, b, warp_specialize: bool):
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # TMA 描述符需要全局内存分配
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    # Hopper warpspec 与 flatten 不兼容
    flatten = False if (warp_specialize and config.is_hopper()) else True
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_descriptor_persistent[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=warp_specialize,  #
        FLATTEN=flatten,
    )
    return c
