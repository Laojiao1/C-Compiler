import sys
import os
from setuptools import setup, Extension
import pybind11

# 注入参数
if len(sys.argv) == 1:
    sys.argv.extend(["build_ext", "--inplace", "--force"])

# 定位当前文件所在的绝对目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将 pybind_ops.cpp 拼接成绝对路径
# cpp_source_absolute = os.path.join(current_dir, "05_1_pybind_ops.cpp")
cpp_source_absolute = os.path.join(current_dir, "05_3_pybind_ops_RAII.cpp")

# 定义 C++ 扩展模块
ext_modules = [
    Extension(
        "my_ops", # 模块名称
        [cpp_source_absolute], # C++ 源文件列表
        # 指定 pybind11 的头文件路径,等价于：E:/conda_envs/pytorch/lib/site-packages/pybind11/include
        include_dirs=[pybind11.get_include()], 
        language="c++", # 指定编译语言为 C++
    ),
]

# 调用 setup() 函数来构建 C++ 扩展模块
setup(
    name="my_ops", # 模块名称
    ext_modules=ext_modules, # 扩展模块列表
)