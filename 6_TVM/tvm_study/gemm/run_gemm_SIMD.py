import tvm
from tvm import te
import numpy as np

# 1. 声明数学表达式 (TE 层)
M, N, K = 1024, 1024, 1024
A = te.placeholder((M, K), name="A", dtype="float32")
B = te.placeholder((K, N), name="B", dtype="float32")
k = te.reduce_axis((0, K), name="k")
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

# 2. 默认调度表
s = te.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 32, 32)
k = C.op.reduce_axis[0]
s[C].reorder(xo, yo, k, xi, yi)

# ─── 优化 3：内层循环向量化 (Vectorize) ───
# 针对硬件的向量执行单元，将最内层的 32 次迭代切出 8 个通道进行向量化
y_outer, y_inner = s[C].split(yi, factor=8)
s[C].vectorize(y_inner)

# ─── 优化 4：局部循环展开 (Unroll) ───
# 将较小的 xi 循环物理展开，消除条件跳转分支
s[C].unroll(xi)

# 降级转换为初始 TensorIR
ir_mod = tvm.lower(s, [A, B, C], simple_mode=True)
print(ir_mod)

# 3. 目标转换 (LLVM 编译)
rt_mod = tvm.build(s, [A, B, C], target="llvm", name="my_gemm")

# 4. Runtime 掐表执行
dev = tvm.cpu(0)
a_tvm = tvm.nd.array(np.random.uniform(size=(M, K)).astype("float32"), dev)
b_tvm = tvm.nd.array(np.random.uniform(size=(K, N)).astype("float32"), dev)
c_tvm = tvm.nd.array(np.zeros((M, N), dtype="float32"), dev)

rt_mod["my_gemm"](a_tvm, b_tvm, c_tvm)

evaluator = rt_mod.time_evaluator("my_gemm", dev, number=10)
gflops = (2 * M * N * K) / evaluator(a_tvm, b_tvm, c_tvm).mean / 1e9
print(f"当前 SIMD 版本性能: {gflops:.2f} GFLOPS")