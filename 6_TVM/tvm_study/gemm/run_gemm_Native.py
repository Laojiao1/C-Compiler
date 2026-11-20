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
print(f"当前 Naive 版本性能: {gflops:.2f} GFLOPS")