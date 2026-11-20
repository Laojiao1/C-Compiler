"""
一个标准的 TVM 编译优化流程：描述（TVMScript）-> 优化（Schedule）-> 编译（build）-> 执行（Runtime）
"""
import numpy as np
import tvm
from tvm.tirx import script as T


@T.prim_func # prim 原语：告诉 tvm 把下面这个函数当作底层的计算内核 kernel 来解析
def add(
    A: T.Buffer((5,), "float32"), # 声明输入和输出的缓冲区，这里(5,)表示为一维数组，长度为 5，数据类型为float32
    B: T.Buffer((5,), "float32"),
    C: T.Buffer((5,), "float32"), # 输出也要作为参数传进去，目的是提前分配好内存，算完直接往里填
):
    T.func_attr({"global_symbol": "add"}) # 给函数贴个标签，指定它在编译后的底层模块里的公开函数名叫 "add"，便于调用

    for i in range(5):
        C[i] = A[i] + B[i] 


# 1. 查看 TensorIR
print(add.script()) # 在真实的深度学习编译中，工程师会重点对这个 TensorIR 做优化（比如循环拆分，向量化，多线程并行），通过 script() 来检查改的对不对

# 2. 编译到 CPU
mod = tvm.build(add, target="llvm") # 如果要编译到 NVIDIA，target="cuda"，编译完成后返回一个 mod（运行时模块），可以理解为一个在内存里生成的动态链接库（类似.so/.dll）

# 3. 准备 Runtime 张量
dev = tvm.cpu(0)

a_np = np.arange(1, 6, dtype=np.float32) # 生成 [1.0, 2.0, 3.0, 4.0, 5.0]
b_np = a_np * 10 # 生成 [10.0, 20.0, 30.0, 40.0, 50.0]

a = tvm.runtime.tensor(a_np, device=dev)
b = tvm.runtime.tensor(b_np, device=dev)
c = tvm.runtime.empty((5,), "float32", device=dev)

# 4. 执行编译后的函数
mod["add"](a, b, c) # 从编译好的模块 mod 中取出名为 "add" 的函数，把 a, b, c 传进去，触发底层机器码的高效执行

# 5. 正确性验证
print("Result:", c.numpy()) # 把 tvm 张量 c 转化为普通的 numpy 数组
np.testing.assert_array_equal(c.numpy(), a_np + b_np) # 做对比，只有全部对的上，程序才会往下走

print("Correctness: PASS")