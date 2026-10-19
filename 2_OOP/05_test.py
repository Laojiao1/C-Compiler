import my_ops

# 1. 查看你的模块和函数文档（可选，但能证明绑定成功）
print("模块文档:", my_ops.__doc__)
print("函数文档:", my_ops.fast_add.__doc__)

# 2. 调用你在 C++ 里亲手写的算子
result = my_ops.fast_add(1.5, 2.0)
print("C++ 算子计算结果:", result)