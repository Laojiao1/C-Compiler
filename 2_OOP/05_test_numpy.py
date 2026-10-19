import numpy as np
import my_ops

# 创建一个 32 位浮点型 numpy 数组
arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)

print("Before C++ Call:", arr)

# 调用 C++ 算子。注意，我们没有接收任何返回值！
my_ops.double_inplace(arr)

# 打印结果
print("After C++ Call:", arr) 