#include <pybind11/pybind11.h>
// 引入 numpy 支持头文件
#include <pybind11/numpy.h>

namespace py = pybind11;

// 为了让下面的 PYBIND11_MODULE 能顺利编译通过，我们把之前的标量加法函数也保留着
float fast_add(float a, float b) {
    return a + b;
}

// Zero-Copy 算子
// 接收一个 numpy array，在 C++ 层直接就地修改它的物理内存
void double_array_inplace(py::array_t<float> input_array) {
    // 步骤 A：剥开 Python 对象的封装，获取底层 buffer 描述符
    // request() 向 Python 解释器索要该 NumPy 数组的内存元数据
    // buf 里包含了指针、大小、形状、步长等信息
    py::buffer_info buf = input_array.request(); 
    
    // 步骤 B：获取最底层的裸指针（极其危险，也极其高效！）
    float* ptr = static_cast<float*>(buf.ptr);
    size_t size = buf.size;
    
    // 用最原生的 C++ for 循环，把 ptr 指向的数组里每一个元素都乘以 2.0f
    // 直接用 CPU 裸指针寻址，速度狂飙！
    for (size_t i = 0; i < size; ++i) {
        ptr[i] *= 2.0f;
    }
}

PYBIND11_MODULE(my_ops, m) {
    m.doc() = "My hardcore C++ operators plugin"; 

    // 原来的加法
    m.def("add", &fast_add, "A function which adds two numbers");
    
    // 把你的 zero-copy 算子注册暴露给 Python，命名为 "double_inplace"
    m.def("double_inplace", &double_array_inplace, "In-place double a numpy array via zero-copy pointer");
}