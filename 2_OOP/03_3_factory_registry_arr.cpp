#include <iostream>
#include <functional>
#include <array>
#include <stdexcept>
#include <cstdint> 

// ----- 系统规范 -----
enum class DeviceType : uint8_t {
    CPU = 0,
    CUDA = 1,
    NPU = 2,
    MAX_DEVICES // 自动获取枚举的总数（当前等于 3）
};

// 模拟 Tensor 数据结构
struct Tensor {
    DeviceType device;
    float data;
};

using OpFunc = std::function<Tensor(const Tensor&, const Tensor&)>;

// ----- 架构层 -----
// 极速算子注册表 (Fast Registry)
class FastOpRegistry {
private:
    // 淘汰 unordered_map，定义定长的连续数组
    // 采用 static_cast 将 MAX_DEVICES 枚举类型 强制转换为 size_t 类型作为数组长度，并初始化为 nullptr
    std::array<OpFunc, static_cast<size_t>(DeviceType::MAX_DEVICES)> registry_{ nullptr };
    
    FastOpRegistry() = default;

public:
    FastOpRegistry(const FastOpRegistry&) = delete;
    FastOpRegistry& operator=(const FastOpRegistry&) = delete;

    // 获取全局唯一实例的引用
    static FastOpRegistry& get_instance() {
        static FastOpRegistry instance;
        return instance;
    }

    // 注册接口
    void register_op(DeviceType device, OpFunc func) {
        // 利用枚举强转作为数组下标，直接写入，速度达到物理极限
        registry_[static_cast<size_t>(device)] = func;
    }

    //  实现极速分发接口
    OpFunc get_op(DeviceType device) {
        // 直接将枚举类型强转为 size_t 作为数组下标
        size_t index = static_cast<size_t>(device);
        
        // 安全边界检查：防止业务端强转一个非法的非枚举数字进来导致数组越界
        if (index >= registry_.size()) {
            throw std::out_of_range("Device type index out of strict bounds.");
        }
        
        // 提取算子函数指针
        OpFunc op = registry_[index];
        
        // 判定是否为空（未注册）
        if (!op) {
            throw std::runtime_error("Operator not registered for this device type.");
        }
        
        return op;
    }
};

// 保留自动注册助手
class OpRegistrar {
public:
    OpRegistrar(DeviceType device, OpFunc func) {
        FastOpRegistry::get_instance().register_op(device, func);
    }
};

#define REGISTRY_CONCAT_IMPL(a, b) a##b
#define REGISTRY_CONCAT(a, b) REGISTRY_CONCAT_IMPL(a, b)

// 宏 API
#define REGISTER_FAST_OP(device, func) \
    static OpRegistrar REGISTRY_CONCAT(__op_reg_, __COUNTER__)(device, func);


// ----- 业务层测试 ----- 
Tensor add_cpu(const Tensor& a, const Tensor& b) {
    std::cout << "[Fast Backend] CPU Array Dispatch Success!" << std::endl;
    return Tensor{DeviceType::CPU, a.data + b.data};
}

// 自动注册
REGISTER_FAST_OP(DeviceType::CPU, add_cpu);

int main() {
    Tensor a{ DeviceType::CPU, 10.0f };
    Tensor b{ DeviceType::CPU, 20.0f };

    try {
        // 主框架的极速调用
        auto op = FastOpRegistry::get_instance().get_op(a.device);
        Tensor result = op(a, b);
        std::cout << "Result: " << result.data << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
    }

    return 0;
}