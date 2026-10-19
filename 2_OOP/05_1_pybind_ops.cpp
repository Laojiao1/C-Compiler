// 引入 pybind11 核心头文件
#include <pybind11/pybind11.h>

// 给 pybind11 起个简短的命名空间别名，要养成良好习惯
namespace py = pybind11;

// 这是一个极其纯粹的 C++ 函数，它甚至不知道 Python 的存在
float fast_add(float a, float b) {
    return a + b;
}

// 定义一个宏：PYBIND11_MODULE
// "my_ops" 是编译后暴露给 Python 的模块名 (类似 import my_ops)
// "m" 是这个模块的对象引用
PYBIND11_MODULE(my_ops, m) {
    // 给模块写个文档字符串（Python 里的 __doc__）
    m.doc() = "My hardcore C++ operators plugin"; 

    // 把底层的 fast_add 函数暴露给 Python
    // m.def("在Python里叫什么名字", &C++函数的指针, "文档说明");
    m.def("fast_add", &fast_add, "A fast addition function");
}