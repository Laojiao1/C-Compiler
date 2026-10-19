#include <iostream>
#include <vector>
#include <cstdint>
#include <type_traits>

// ----- 函数模板 -----
// 将原本写死的 float 改写为泛型占位符 T
template <typename T>
void vector_add(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& out) {
    // 静态断言，确保 T 是一个数学运算类型 (int, float, double 等)
    static_assert(std::is_arithmetic<T>::value, "vector_add 仅支持数学运算类型 (int, float, double 等) 的加法操作！");

    std::cout << "[Generic Backend] Executing standard for-loop addition..." << std::endl;
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
    }
}

// 模板全特化 (Template Specialization)：针对 int8_t 数据类型，彻底绕开通用逻辑，实现硬件级特殊加速
template <>
void vector_add<int8_t>(const std::vector<int8_t>& a, const std::vector<int8_t>& b, std::vector<int8_t>& out) {
    std::cout << "[SIMD Accelerated] Executing INT8 specific addition..." << std::endl;
    // 模拟硬件加速逻辑
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i]; // 实际上这里可以是 SIMD 指令调用
    }
}

// ----- 测试 -----
int main() {
    // 场景 A：常规浮点运算
    std::vector<float> a_f = {1.1f, 2.2f};
    std::vector<float> b_f = {3.3f, 4.4f};
    std::vector<float> out_f(2);
    
    // 调用模板函数，执行 float 类型的加法
    vector_add(a_f, b_f, out_f);
    
    
    // 场景 B：量化推理运算
    std::vector<int8_t> a_i8 = {1, 2};
    std::vector<int8_t> b_i8 = {3, 4};
    std::vector<int8_t> out_i8(2);
    
    // 调用模板函数，执行 int8_t 类型的加法
    vector_add(a_i8, b_i8, out_i8);
    
    return 0;
}