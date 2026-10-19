#include <pybind11/pybind11.h>
// 引入 numpy 支持头文件
#include <pybind11/numpy.h>
#include <thread>
#include <chrono>
#include <iostream>

namespace py = pybind11;

// 解决 Python GIL 的 RAII 版本
void slow_add_array_inplace(py::array_t<float> input_array) {
    // 1. 获取物理指针（这部分还涉及到 Python 对象操作，所以必须带锁执行）
    py::buffer_info buf = input_array.request();
    float* ptr = static_cast<float*>(buf.ptr);
    size_t size = buf.size;

    // 2. 进入纯物理内存的计算区域！不再需要请求 Python API 了！
    {
        // 实例化 RAII GIL 释放器，让 Python 其他线程解冻！
        py::gil_scoped_release release;
        
        std::cout << "[C++ Thread] GIL Released! Starting heavy computation (3 seconds)..." << std::endl;
        
        // 模拟底层的重度计算，故意挂起 3 秒钟
        std::this_thread::sleep_for(std::chrono::seconds(3));
        
        for (size_t i = 0; i < size; ++i) {
            ptr[i] = ptr[i] + 10.0f;
        }
        
        std::cout << "[C++ Thread] Computation done. RAII will now automatically re-acquire GIL." << std::endl;
        
    } // 注意这个右大括号！生命周期结束，release 对象的析构函数被调用，GIL 被重新获取。
}


// 为了让下面的 PYBIND11_MODULE 能顺利编译通过，我们把之前的标量加法函数也保留着
float fast_add(float a, float b) {
    return a + b;
}

// Zero-Copy 算子
// 接收一个 numpy array，在 C++ 层直接就地修改它的物理内存
void double_array_inplace(py::array_t<float> input_array) {
    // 步骤 A：剥开 Python 对象的封装，获取底层 buffer 描述符
    py::buffer_info buf = input_array.request();
    
    // 步骤 B：获取最底层的裸指针（极其危险，也极其高效！）
    float* ptr = static_cast<float*>(buf.ptr);
    size_t size = buf.size;
    
    // 用最原生的 C++ for 循环，把 ptr 指向的数组里每一个元素都乘以 2.0f
    // 直接用 CPU 裸指针寻址，提升速度
    for (size_t i = 0; i < size; ++i) {
        ptr[i] *= 2.0f;
    }
}

PYBIND11_MODULE(my_ops, m) {
    m.doc() = "My hardcore C++ operators plugin"; 

    // 保留之前的算子
    m.def("add", &fast_add, "A function which adds two numbers");
    m.def("double_inplace", &double_array_inplace, "In-place double a numpy array via zero-copy pointer");

    // 新算子
    m.def("slow_inplace", &slow_add_array_inplace, "Slow in-place addition releasing GIL");
}