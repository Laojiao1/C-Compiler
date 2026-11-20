import tvm
from tvm import te
import numpy as np

# ===============================================================
# 1. 声明关键数据结构（Compute：我们要算什么）
# ===============================================================
M, N, K = 1024, 1024, 1024

# 定义输入矩阵 A 和 B 的占位符
A = te.placeholder((M, K), name="A", dtype="float32")
B = te.placeholder((K, N), name="B", dtype="float32")
k = te.reduce_axis((0, K), name="k")

# 描述通用矩阵乘法 (GEMM) 的数学公式
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

# ===============================================================
# 2. 转换与下沉（向量化 + 循环展开全面压榨）
# ===============================================================
s = te.create_schedule(C.op)

# ─── 1. 执行 32x32 切块 ───
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 32, 32)
k = C.op.reduce_axis[0]

# ─── 2. 重新排列轴 ───
s[C].reorder(xo, yo, k, xi, yi)

# ─── 3. 大招：最内层循环(yi)开启向量化并行 ───
# 假设你的 CPU 至少支持 AVX2（256位带宽），单周期可并发执行 8 个 float32 计算
# 我们可以把内层的 32 次迭代，继续切出 8 个元素进行向量化加速！
y_outer, y_inner = s[C].split(yi, factor=8)
s[C].vectorize(y_inner)

# ─── 4. 辅助大招：将倒数第二层小循环(xi)直接物理展开 ───
# 消除循环分支判断，让 CPU 疯狂吃连续指令
s[C].unroll(xi)

# ─── 5. 把最外层的行循环(xo)直接扔给多核并行 ───
# TVM 会自动在底层帮我们生成多线程并行的 OpenMP 机器码
s[C].parallel(xo)

# 再次转换为 IRModule，观察终极形态的 TensorIR
ir_mod = tvm.lower(s, [A, B, C], simple_mode=True)

print("\n" + "="*50)
print(" 优化后的 IRModule 内部（带 *vectorized* 标记的 TensorIR）:")
print("="*50)
print(ir_mod) # 打印它！看看底层指针发生了何种质变

# ===============================================================
# 3. 目标转换（Target Translation：交棒给大 Boss LLVM）
# ===============================================================
target = "llvm"
# 启动 Target Translation 阶段，调用 LLVM 将上面的 TensorIR 循环彻底翻译成你当前 CPU 认的二进制机器码
rt_mod = tvm.build(s, [A, B, C], target=target, name="my_gemm")

# ===============================================================
# 4. Runtime 执行（Execution：用 PackedFunc 轰出真实跑分）
# ===============================================================
dev = tvm.cpu(0)

# 准备测试数据（用 NumPy 生成 float32 随机数）
a_np = np.random.uniform(size=(M, K)).astype(A.dtype)
b_np = np.random.uniform(size=(K, N)).astype(B.dtype)
c_np = np.zeros((M, N), dtype=C.dtype)

# 将数据从主机搬运到 TVM Runtime 的 Tensor 对象中
a_tvm = tvm.nd.array(a_np, dev)
b_tvm = tvm.nd.array(b_np, dev)
c_tvm = tvm.nd.array(c_np, dev)

# 从 rt_mod (Module) 里，按名字捞出抹平了类型的 PackedFunc
gemm_packed_func = rt_mod["my_gemm"]

# 踩下油门！通过 PackedFunc 的标准化接口，直接驱动底层的机器码运行
gemm_packed_func(a_tvm, b_tvm, c_tvm)

print("\n" + "="*50)
print(" Runtime 执行阶段：计算完成！")
print("="*50)
np.testing.assert_allclose(c_tvm.numpy(), np.dot(a_np, b_np), rtol=1e-5)
print("✅ 结果比对完全正确！TVM 编译的二进制文件成功在硬件上输出了正确答案！")

# 评估未调优版本的真实硬件跑分（GFLOPS）
evaluator = rt_mod.time_evaluator("my_gemm", dev, number=10)
mean_time = evaluator(a_tvm, b_tvm, c_tvm).mean
gflops = (2 * M * N * K) / mean_time / 1e9
print(f"当前版本的 GEMM 吞吐量性能: {gflops:.2f} GFLOPS")