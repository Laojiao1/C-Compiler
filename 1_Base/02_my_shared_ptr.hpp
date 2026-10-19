#pragma once

#include <utility>
#include <cstddef>

namespace ai_infra {
    
template <typename T> class shared_ptr {
private:
    T *ptr_; // 指向真实数据的裸指针
    size_t *ref_count_; // 指向堆上分配的“控制块计数器”
    
    // 资源释放
    void release_resource() {
        if(ref_count_) {
            (*ref_count_)--; // 计数器-1
            
            if(*ref_count_ == 0) { // 如果是最后一个
                // 释放数据
                delete ptr_;
                delete ref_count_;
                
                // 一定要记得置空！防止野指针！
                ptr_ = nullptr;
                ref_count_ = nullptr;
            }
        }
    }

public:
    // ----- 构造与析构 -----
    // 构造函数初始化
    constexpr shared_ptr() noexcept : ptr_(nullptr),
    								ref_count_(nullptr) {}
    
    // 构造函数，接管裸指针
    explicit shared_ptr(T *p) : ptr_(p) {
        if(p) {
            // 如果有数据在堆上，new一个控制块，并初始化计数器为1
            ref_count_ = new size_t(1);
        } else {
            ref_count_ = nullptr;
        }
    }
    
    // 析构函数
    ~shared_ptr() {
        release_resource();
    }
    
    // ----- 拷贝语义 -----
    shared_ptr(const shared_ptr &other) noexcept :
    	ptr_(other.ptr_), ref_count_(other.ref_count_) {
            if(ref_count_) {
                (*ref_count_)++;
            }
    }
    
    // 重构拷贝赋值运算符
    shared_ptr &operator=(const shared_ptr &other) noexcept {
        // 防止自赋值
        if(this != &other) {
            // 先释放当前资源
            release_resource();
            
            // 然后接管 other 的资源
            ptr_ = other.ptr_;
            ref_count_ = other.ref_count_;
            
            // 对应新控制块的计数值 + 1
          	if (ref_count_) {
            	(*ref_count_)++;
          	}
        }
        return *this;
    }
    
    // ----- 移动语义 -----
    // 移动构造函数（单纯的窃取，不需要操作 ref_count_ 里的数字）
    shared_ptr(shared_ptr &&other) noexcept
       	: ptr_(std::exchange(other.ptr_, nullptr)),
    		ref_count_(std::exchange(other.ref_count_, nullptr)){}
    
    // 移动赋值运算符
    shared_ptr &operator=(shared_ptr &&other) noexcept {
        if (this != &other) {
            // 做法同上，注意这里不需要处理 ref_count_
      		release_resource(); 
      		ptr_ = std::exchange(other.ptr_, nullptr);
      		ref_count_ = std::exchange(other.ref_count_, nullptr);
    	}
		return *this;
    }
    
    // ----- 一些辅助操作 -----
    // 获取引用数
    size_t use_count() const noexcept {
        return ref_count_ ? *ref_count_ : 0;
    }
    
    // 重载运算符
    T &operator*() const { return *ptr_; }
    T *operator->() const { return ptr_; }
    
    // 获取当前指针
    T *get() const { return ptr_; }
};
    
} // namespace ai_infra