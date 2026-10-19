#include <iostream>
#include <cstring>
#include <type_traits>
#include <cstdint>

// ----- 架构层：底层泛型张量类 (Class Template) ----- 
// 声明类模板
// T 是【类型模板参数】，N 是【非类型模板参数】（必须是一个常数/整型）
template <typename T, size_t N>
class StaticTensor {
private:
    // 定义原生数组
    // 利用编译期就确定的常数 N 直接开辟连续的物理内存，拒绝一切堆内存分配
    T data_[N];
    
public:
    // 构造函数：初始化内存
    StaticTensor() {
        std::memset(data_, 0, sizeof(data_));
    }

    //  实现一个获取大小的接口
    // 因为 N 在编译期已经是已知常量，这里甚至不需要读取内存，直接返回 N
    size_t size() const {
        return N;
    }

    // 实现通用加法算子
    StaticTensor<T, N> add(const StaticTensor<T, N>& other) const {
        StaticTensor<T, N> result;
        for (size_t i = 0; i < N; ++i) {
            result.data_[i] = this->data_[i] + other.data_[i];
        }
        return result;
    }
};

// ----- 业务层测试 -----
int main() {
    // 场景 A：实例化一个装有 128 个 float 的张量
    StaticTensor<float, 128> tensor_a;
    std::cout << "Tensor A size: " << tensor_a.size() << " elements." << std::endl;

    // 场景 B：实例化一个装有 64 个 int8_t 的张量
    StaticTensor<int8_t, 64> tensor_b;
    std::cout << "Tensor B size: " << tensor_b.size() << " elements." << std::endl;

    // 破坏性测试猜想验证：
    // 如果写下这一行：tensor_a.add(tensor_b);
    // 编译器会直接报错：no matching function for call to...
    // 甚至连运行的机会都不给，完美封死所有维度错配的可能！

    return 0;
}