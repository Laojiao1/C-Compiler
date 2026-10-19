#pragma once

#include <utility> // for std::move, std::exchange

namespace ai_infra {

// 模板，代表这个删除器可以处理任意类型 T 的指针
template <typename T> 
struct DefaultDeleter { // 默认的 Deleter，后续可以根据需要进行扩展
    void operator()(T* ptr) const { // 重载括号运算符，以后可以直接使用 DefaultDeleter<int> del; del(ptr);
        if(ptr) {
            delete ptr; // delete：1.调用析构函数（如果有）2. free
        }
    }
};
    
// 定义带有自定义 Deleter 模板参数的 unique_ptr
template <typename T, typename Deleter = DefaultDeleter<T>>
class unique_ptr {
private:
    T* ptr_; // 裸指针
    Deleter deleter_; // 删除器实例
    
public:
    // ----- 构造与析构 -----
    // constexpr：表示在编译期就能算出来，用于极致优化
    // noexcept：向编译器保证这个函数绝对不会抛出异常。编译器看到它，就会省去很多生成异常处理栈展开的底层代码
    constexpr unique_ptr() noexcept : ptr_(nullptr) {}
    
    // 显式指针构造函数，用于获取资源
    explicit unique_ptr(T* p) noexcept : ptr_(p) {}
    
    // 析构函数
    ~unique_ptr() {
        deleter_(ptr_); // 利用 deleter_ 安全释放 ptr_
    }
    
    // ----- 独占所有权 -----
    // 显式禁止 “拷贝构造函数” 和 “拷贝赋值运算符”，防止重复释放
    unique_ptr(const unique_ptr& other) = delete;
    unique_ptr& operator=(const unique_ptr& other) = delete;

	// ----- 资源转换（移动语义）-----
    // 这里的 other 代表一个“即将消亡的值”（比如临时变量，或者被 std::move 强制转换的变量）
    // std::exchange(a, b)：把 a 的旧值返回，然后把 b 赋值给 a。在这里，它等价于：this->ptr_ = other.ptr_; other.ptr_ = nullptr;
    unique_ptr(unique_ptr&& other) noexcept 
    : ptr_(std::exchange(other.ptr_, nullptr)), 
      deleter_(std::move(other.deleter_)) {}
    
    // 移动赋值运算符
    // 注意逻辑：先释放自己原有的资源，再去偷 other 的资源
    unique_ptr& operator=(unique_ptr&& other) noexcept {
        if(this != &other) { // 防止自赋值
            deleter_(ptr_); // 先释放自己的资源
            // 偷 other 资源
            ptr_ = std::exchange(other.ptr_, nullptr);
            deleter_ = std::move(other.deleter_);
        }
        return *this;
    }
    
    // ----- 底层指针操作 -----
    // 重载解引用，目的是为了让它用起来像原生指针一样
    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    
    // 获取底层裸指针
    T* get() const { return ptr_; }
    
    // 实现 reset 逻辑：注意要先释放旧资源，然后接管新指针
    void reset(T* p = nullptr) noexcept {
        T* old_ptr = std::exchange(ptr_, p);
        deleter_(old_ptr);
    } // 这里不直接 deleter_(ptr_); ptr_ = p; 的原因是为了防止重置同一个指针（如 p1.reset(p1.get())）导致先被释放的 Bug
    
    // 实现 release 逻辑：不再管理该资源，由调用者手动 delete
    T* release() noexcept {
        // 放弃所有权，拿走当前指针，内部置为空，并把拿走的指针返回给外界
        return std::exchange(ptr_, nullptr);
    }
};
    
} // namespace ai_infra