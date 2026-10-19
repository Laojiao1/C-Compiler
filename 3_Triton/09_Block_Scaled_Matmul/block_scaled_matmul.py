# Scale preshuffling on AMD GPUs
#
# 与 NVIDIA GPU 类似，在采用 CDNA4 架构的 AMD GPU 上，缩放 MFMA 指令原生支持缩放矩阵乘法。
# 由于它只支持 OCP 微缩放格式，因此每个缩放都是一个 8 位值，它缩放 A 或 B 操作数张量的 32 个元素。
# 尺度以 8 位张量的形式存储。由于 MFMA 指令是线程束级指令，这意味着每个线程都会向 MFMA 指令提供一组固定的操作数值。
#
# 例如，在形状为 16x16x128 的 MFMA 指令中：
# - 4 个线程沿 K 维度贡献元素。
# - 16 个线程沿 M 或 N 维度贡献元素。
#
# 从尺度张量的角度来看，即使 K 维在共享内存中连续存储，由于与其他线程交错，每个线程看到的沿 K 维的元素也是步长的。
# 他的步幅限制了使用向量化内存访问加载比例值的能力。
#
# 我们的目标是重新组织尺度张量，使得：
# 1. 每个线程将 4 个 MFMA 操作所需的 4 个缩放值存储在连续内存中。
# 2. 连续线程访问连续内存位置，在绕过 LDS 时改善全局内存合并，这对于“瘦”矩阵乘法尤其有利。
#
# 我们考虑两种 MFMA 情况：一种是非 K 维数为 16，另一种是非 K 维数为 32。
# 两者的最小预洗牌尺寸均为 32x32x256。
# 例如，对于 32x256 的操作数块，相应的尺度张量形状为 32x8，其中每个尺度沿 K 维度覆盖 32 个元素。
#
# 每个线程为每次 MFMA 操作保存一个标度。我们将 4 个标度值（对应 4 次不同的 MFMA 操作）在内存中并排存储。
#
# Case 1: mfma_scaled_16x16x128
#
# 包装顺序: mfma_op_0, mfma_op_2, mfma_op_1, mfma_op_3
#
#            K = 128       K = 128
#        +------------+ +------------+
#    M=16|  MFMA op 0 | |  MFMA op 1 |
#        +------------+ +------------+
#    M=16|  MFMA op 2 | |  MFMA op 3 |
#        +------------+ +------------+
#
# Case 2: mfma_scaled_32x32x64
#
# 包装顺序: mfma_op_0, mfma_op_1, mfma_op_2, mfma_op_3
#
#            K=64     K=64     K=64     K=64
#        +--------+ +--------+ +--------+ +--------+
#    M=32| op 0   | | op 1   | | op 2   | | op 3   |
#        +--------+ +--------+ +--------+ +--------+

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

"""
该函数是 Triton 内核的启动元数据钩子，用于在运行时动态生成带具体精度格式与矩阵尺寸的内核名称，并计算理论浮点运算量（FLOPs），以供性能分析器（Proton Profiler）统计耗时与计算吞吐（TFLOP/s）。
"""
def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    kernel_name = kernel.name
    if "ELEM_PER_BYTE_A" and "ELEM_PER_BYTE_B" and "VEC_SIZE" in args:
        if args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 1:
            kernel_name += "_mxfp8"
        elif args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 2:
            kernel_name += "_mixed"
        elif args["ELEM_PER_BYTE_A"] == 2 and args["ELEM_PER_BYTE_B"] == 2:
            if args["VEC_SIZE"] == 16:
                kernel_name += "_nvfp4"
            elif args["VEC_SIZE"] == 32:
                kernel_name += "_mxfp4"
    ret["name"] = f"{kernel_name} [M={M}, N={N}, K={K}]"
    ret["flops"] = 2.0 * M * N * K
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_matmul_kernel(
    # 硬件 TMA 描述符（封装了显存地址、多维尺寸与步长
    a_desc, a_scale_desc,
    b_desc, b_scale_desc,
    c_desc,
    # 矩阵形状与常量
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    output_type: tl.constexpr,  # 0: FP32, 1: FP16, 2: FP8
    ELEM_PER_BYTE_A: tl.constexpr,  # FP4为2（1字节装2个元素），FP8为1
    VEC_SIZE: tl.constexpr, # NVFP4 为 16，MXFP4 为 32
    BLOCK_M: tl.constexpr, # 每个 Block 处理的 M 切片大小（如128）
    BLOCK_N: tl.constexpr, # 每个 Block 处理的 N 切片大小（如256）
    BLOCK_K: tl.constexpr, # K 方向每次迭代步长（如256）
    # 宏块倍数，用于匹配硬件 5D TMA 描述符的索引粒度
    rep_m: tl.constexpr,  # M 维度相对于硬件基础宏块（128 元素）的倍数
    rep_n: tl.constexpr,  # N 维度相对于硬件基础宏块的倍数
    rep_k: tl.constexpr,  # 沿 K 维 Scale 块的宏块数量
    NUM_STAGES: tl.constexpr,  # 软件流水线级数
    # 控制编译器是否对累加器寄存器进行多缓冲。
    # 在 Blackwell 等架构上用于精细平衡寄存器占用，防止溢出
    disallow_acc_multi_buffer: tl.constexpr,
):
    # ---------------- 步骤 A: 确定输出类型与坐标定位 ----------------
    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.float8e4nv

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)

    # 列优先网格映射：先沿 M 轴纵向遍历，再切到下一列（最大化复用 B 矩阵的 L2 Cache）
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    # 物理数据与 5D Scale 的初始偏移量
    # 因为 Scale 的 5D 描述符是按基础宏块（128 元素）建维度的，所以这里乘以 rep_m rep_n 得到宏块级别的索引起点
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k_a = 0
    offs_k_b = 0
    offs_scale_m = pid_m * rep_m
    offs_scale_n = pid_n * rep_n
    offs_scale_k = 0

    # 判断是否为 FP8(A) x FP4(B) 的混合精度
    MIXED_PREC: tl.constexpr = (ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2)

    # 累加器始终保持高精度 FP32
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---------------- 步骤 B: 沿 K 轴切片迭代 ----------------
    # 沿 K 轴以 BLOCK_K 为 步长遍历
    # num_stages 开启软件流水线（Prefetch 流水线）
    for k in tl.range(
        0, tl.cdiv(K, BLOCK_K),
        num_stages=NUM_STAGES,
        disallow_acc_multi_buffer=disallow_acc_multi_buffer):

        # 1. TMA 硬件异步加载输入数据
        a = a_desc.load([offs_am, offs_k_a])
        b = b_desc.load([offs_bn, offs_k_b])

        # 2. TMA 加载 5D 打包的 Scale 块 (一次拉取大块数据，避免显存拥堵)
        scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
        scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])

        # 3. 寄存器内维度重整：将 5D 打包数据解开并对齐为 2D 逻辑 Scale 矩阵
        """
        reshape(rep_m, rep_k, 32, 4, 4)：将拉入的扁平数据按底层存储的子块划分
        trans(0, 3, 2, 1, 4)：在寄存器内部完成维度转置，调整子块轴的物理排布
        reshape(BLOCK_M, BLOCK_K // VEC_SIZE)：最终合并还原为与 A 严格对应的 2D 形状，使每一行沿 K 轴每 VEC_SIZE 个元素恰好对应 1 个 8-bit Scale
        最终形状变成: (BLOCK_M, BLOCK_K // VEC_SIZE)
        """
        scale_a = scale_a.reshape(rep_m, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_b = scale_b.reshape(rep_n, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

        # 4. 硬件级微缩放 MMA（直接调用底层 Tensor Core 融合计算）
        if MIXED_PREC:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator)
        elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)
        else:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)

        # 5. 指针沿 K 轴推进
        offs_k_a += BLOCK_K // ELEM_PER_BYTE_A
        offs_k_b += BLOCK_K // ELEM_PER_BYTE_B
        offs_scale_k += rep_k

    # ---------------- 步骤 C: 结果写回 ----------------
    # 将累加器转为目标类型，通过 2D TMA 直接异步刷回显存
    c_desc.store([offs_am, offs_bn], accumulator.to(output_dtype))


def block_scaled_matmul(a_desc, a_scale_desc, b_desc, b_scale_desc, dtype_dst, M, N, K, rep_m, rep_n, rep_k, configs,
                        clc=False):
    output = torch.empty((M, N), dtype=dtype_dst, device="cuda")
    if dtype_dst == torch.float32:
        dtype_dst = 0
    elif dtype_dst == torch.float16:
        dtype_dst = 1
    elif dtype_dst == torch.float8_e4m3fn:
        dtype_dst = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype_dst}")

    BLOCK_M = configs["BLOCK_SIZE_M"]
    BLOCK_N = configs["BLOCK_SIZE_N"]
    c_desc = TensorDescriptor.from_tensor(output, [BLOCK_M, BLOCK_N])

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    num_stages = configs["num_stages"]
    if clc:
        num_stages = min(num_stages, 3)
    block_scaled_matmul_kernel[grid](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        M,
        N,
        K,
        dtype_dst,
        configs["ELEM_PER_BYTE_A"],
        configs["ELEM_PER_BYTE_B"],
        configs["VEC_SIZE"],
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_N"],
        configs["BLOCK_SIZE_K"],
        rep_m,
        rep_n,
        rep_k,
        num_stages,
        disallow_acc_multi_buffer=configs["disallow_acc_multi_buffer"],
        clc=clc,
    )
    return output