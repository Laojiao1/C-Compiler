#include <iostream>
#include <string>
#include <unordered_map>
#include <functional>
#include <stdexcept>

// 模拟 Tensor 数据结构
struct Tensor {
    std::string device; // "CPU" or "CUDA"
    std::string dtype;  // "FP32" or "INT8"
    float data;         // 简化版，实际为数据指针
};

// 定义底层算子函数的统一签名
using OpFunc = std::function<Tensor(const Tensor&, const Tensor&)>;

// ----- 架构层 -----

// 全局算子注册表 (Registry)：采用经典的 Meyers 单例模式
class OpRegistry {
private:
    // 定义哈希表 registry_：
    // Key 是字符串（描述算子的特征签名，例如 "add_CPU_FP32"）
    // Value 是对应的函数指针 OpFunc
    std::unordered_map<std::string, OpFunc> registry_;

    // 私有化构造函数，阻止外部 new 实例化
    OpRegistry() = default;

public:
    // 禁用拷贝和赋值操作，防止破坏单例，严防多重注册表导致的数据不一致
    OpRegistry(const OpRegistry&) = delete;
    OpRegistry& operator=(const OpRegistry&) = delete;

    // 获取全局唯一实例的引用
    static OpRegistry& get_instance() {
        // C++11 标准保证了局部静态变量初始化的线程安全性
        static OpRegistry instance;
        return instance;
    }

    // 实现注册接口
    // 将传入的 name 和 func 存入 registry_ 中
    void register_op(const std::string& name, OpFunc func) {
        registry_[name] = func;
    }

    // 实现分发接口：根据 name 去 registry_ 里查找对应的函数并返回
    OpFunc get_op(const std::string& name) {
        auto it = registry_.find(name); 
        if (it == registry_.end()) { // 如果找不到，抛出 std::runtime_error 异常，并带上出错的 name 提示。
            throw std::runtime_error("Operator not found: " + name);
        }

        return it->second;
    }
};

// 自动注册助手类 (Registrar)：将函数的注册动作封装到对象构造阶段，利用 C++ 静态变量初始化期发起注入
class OpRegistrar {
public:
    // 构造函数：在这个构造函数内部获取单例，并将 name 和 func 注册进去
    OpRegistrar(const std::string& name, OpFunc func) {
        OpRegistry::get_instance().register_op(name, func);
    }
};

// ----- 工作区 ----- 
// 以下代码完全可以写在一个完全独立的 add_cpu.cpp 中，他们既不包含也不修改 main.cpp，完全实现了代码解耦

// 自定义底层硬件算子实现
Tensor add_cpu_fp32(const Tensor& a, const Tensor& b) {
    std::cout << "[Backend: CPU FP32] Executing Add (Mem Addr: 0x...) " << std::endl;
    return Tensor{"CPU", "FP32", a.data + b.data};
}

Tensor add_cuda_fp32(const Tensor& a, const Tensor& b) {
    std::cout << "[Backend: CUDA FP32] Executing Add..." << std::endl;
    return Tensor{"CUDA", "FP32", a.data + b.data};
}


// 采用宏定义注册API，自动生成变量名，杜绝全局命名空间污染
// 注意：采用双层宏展开的方式，确保 __COUNTER__ 能够被正确展开为整数
#define REGISTRY_CONCAT_IMPL(a, b) a##b
#define REGISTRY_CONCAT(a, b) REGISTRY_CONCAT_IMPL(a, b)

// 完成注册宏的定义
// 利用 REGISTRY_CONCAT 把 "__op_reg_" 和 __COUNTER__ 拼起来作为隐藏变量名
// 预期宏展开结果：static OpRegistrar __op_reg_0(name, func);
#define REGISTER_OP(name, func) \
    static OpRegistrar REGISTRY_CONCAT(__op_reg_, __COUNTER__)(name, func);


// 用宏进行注册
// 编译器在后面会处理成：static OpRegistrar REGISTRY_CONCAT(__op_reg_, __COUNTER__)("add_CPU_FP32", add_cpu_fp32);
REGISTER_OP("add_CPU_FP32", add_cpu_fp32); 
REGISTER_OP("add_CUDA_FP32", add_cuda_fp32);

// 主函数
int main() {
    std::cout << "--- Main System Booting ---" << std::endl;
    
    Tensor a{ "CPU", "FP32", 1.5f };
    Tensor b{ "CPU", "FP32", 2.0f };

    try {
        auto op1 = OpRegistry::get_instance().get_op("add_CPU_FP32");
        auto result1 = op1(a, b);
        std::cout << "Result: " << result1.data << std::endl;
        
        auto op2 = OpRegistry::get_instance().get_op("add_CUDA_FP32");
        auto result2 = op2(a, b);
        std::cout << "Result: " << result2.data << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
    }

    return 0;
}