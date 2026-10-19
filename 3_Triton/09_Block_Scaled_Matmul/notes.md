# 块缩放矩阵乘法 (Block-Scaled GEMM)

块缩放矩阵乘法 (Block-Scaled GEMM) 是下一代低精度计算

## 背景

**现状**：现代大模型（LLM）参数量达数百亿、数千亿（如 LLaMA、DeepSeek）

**问题**：如果使用常规的 **16位浮点数（FP16/BF16）** 存储和计算：

- **显存装不下**：千亿模型仅权重就需要几百 GB 显存
- **带宽跑不赢**：GPU 计算单元常常在“干等”显存把数据搬运过来（Memory-Bound）

为了压缩体积，业内很早就开始使用低精度（4bit，8bit），但遇到了瓶颈：

在传统粗粒度量化中：整个矩阵或整个一行共同一个缩放系数 Scale，但如果这行数据中有一个异常巨大的数（离群值），则会把整个 Scale 拉的极大。这会导致其余正常的小数值在量化为 4-bit 时，全部被强行四舍五入变成了 0，精度彻底崩塌

因此，想要 FP4 的 4 倍显存压缩，模型就变“傻”；想要模型保持聪明，只能退回 FP16，但又会爆显存

----

**解决思路：微缩放 (Microscaling / Block Scaling)**

不让整行共用一个 Scale，而是化整为零：

- 沿着矩阵乘法的累加维度（K），让每 16 个或 32 个元素 组成一个小向量组
- 每个小向量组内拥有自己独立的、专属的 8-bit Scale

```
[ 0.1, 0.3, 0.2, 0.1 ] -> 共享 Scale_1 (针对小数值，精度保留极高)
[ 95.0, 99.0, 88.0, 92.0 ] -> 共享 Scale_2 (针对离群值，单独管理，互不干扰)
```

两种主流行业标准如下：

| 格式名称          | 主导厂商 / 组织                  | 向量分组长度 (VEC_SIZE)   | Scale 精度        |
| ----------------- | -------------------------------- | ------------------------- | ----------------- |
| OCP MXFP4 / MXFP8 | OCP 开放计算组织（AMD CDNA4 等） | **32** 个元素共用 1 Scale | 8-bit (E8M0/E4M3) |
| NVFP4             | NVIDIA（Blackwell / SM100 架构） | **16** 个元素共用 1 Scale | 8-bit (E4M3)      |

---

**现在有了数学思想，但写成 GPU 代码时又会遇到 3 个严重的性能阻碍：**

**难题 1：反量化会拖慢计算**

- 如果靠写  CUDA/Triton 代码把 FP4 乘以 Scale 还原成 FP16 再算乘法，寄存器会瞬间爆炸，速度比直接算 FP16 还慢

解法：采用硬件级融合指令

- NVIDIA Blackwell 提供了 `tcgen05.mma`，AMD CDNA4 提供了 `scaled MFMA`
- Triton 统一封装为 **`tl.dot_scaled(...)`**：硬件在芯片电路内部 **边反量化边算乘加，在一个周期内就计算出结果**

**难题 2：Scale 太小太散，显存搬运堵车**

- 每个 Scale 只有 1 个 Byte（8-bit）。如果让 GPU 每次都去显存捞 16 Bytes 的微小 Scale，显存搬运引擎（TMA）会被海量的零碎请求直接堵死

解法：采用 5D TMA 描述符（NVIDIA）/ 内存预重排（AMD Preshuffling）

- 只需要在显存中把小 Scale 重新组织为 5 维的大数据包（如单次搬运 256 字节），一口气喂给 GPU 即可

**难题 3：打包进来的数据与计算排布对不上**

- 5D 描述符搬进寄存器的数据是为了“搬运吞吐”打包的，无法直接拿来做 $A \times B$ 的矩阵乘法

解法：维度重塑

- 利用 `reshape -> trans -> reshape` 把打包数据还原成标准的 2D 逻辑 Scale 矩阵即可（在这个过程中寄存器零开销）

## 代码实战

```py
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
    BLOCK_M: tl.constexpr, # 每个Block处理的M切片大小（如128）
    BLOCK_N: tl.constexpr, # 每个Block处理的N切片大小（如256）
    BLOCK_K: tl.constexpr, # K方向每次迭代步长（如256）
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
```

