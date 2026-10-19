#include "01_my_unique_ptr.hpp"
#include <iostream>
#include <string>

using namespace std;

// 我们的“靶子”类：模拟深度学习中的 Tensor
class Tensor {
public:
  	string name_;

  	Tensor(string name) : name_(name) {
    	cout << "[Tensor] " << name_ << " 被创建，分配 1GB 显存！" << endl;
  	}

  	~Tensor() {
    	cout << "[Tensor] " << name_ << " 被销毁，释放 1GB 显存！" << endl;
  	}

  	void compute() {
    	cout << "  -> [计算图] " << name_ << " 正在执行矩阵乘法..." << endl;
  	}
};

// 模拟一个算子执行引擎，它要求强制接管 Tensor 的所有权
void run_compute(ai_infra::unique_ptr<Tensor> t) {
  	cout << "  >>> 【算子引擎】成功夺取所有权！开始处理..." << endl;
  	t->compute();
  	cout << "  >>> 【算子引擎】处理完毕，引擎即将关闭，局部变量 t 马上要死亡..." << endl;
} // t 在这里离开作用域，触发 RAII 释放！

int main() {
  	cout << "------------- RAII 自动释放测试 -------------" << endl;
  	{
    	ai_infra::unique_ptr<Tensor> ptr1(new Tensor("Tensor_A"));
    	ptr1->compute();
    	cout << "  >>> 准备离开作用域..." << endl;
  	} // 期待：在这里自动调用 Tensor_A 的析构函数
  	cout << "  >>> 成功离开作用域！(如果你看到上面输出了释放信息，说明 RAII 成功！)"<< endl;

  	cout << "------------- 防 Copy 编译拦截测试 -------------" << endl;
  	ai_infra::unique_ptr<Tensor> ptr2(new Tensor("Tensor_B"));

  	// 【互动环节】：你可以试着把下面这行代码取消注释，然后尝试编译。
  	// 如果你写的 unique_ptr(const unique_ptr&) = delete 生效了，编译器会报错！
  	// ai_infra::unique_ptr<Tensor> ptr3 = ptr2;

  	cout << "  >>> 拷贝行为已被编译器严格拦截（详见代码注释）。" << endl;

  	cout << "------------- Move 所有权转移测试 -------------" << endl;
  	ai_infra::unique_ptr<Tensor> ptr4(new Tensor("Tensor_C"));

  	cout << "  >>> 主函数：准备把 Tensor_C 交给底层算子引擎处理" << endl;
  	// 注意：不能直接传 ptr4，必须用 std::move
  	// 伪装成右值，触发移动构造函数（窃取）！
  	run_compute(std::move(ptr4));

  	cout << "  >>> 主函数：底层算子引擎执行完毕，控制权回到主函数" << endl;

  	// 验证窃取是否成功，原指针是否被置空
  	if (ptr4.get() == nullptr) {
    	cout << " 所有权已交出！ptr4 内部指针已被 std::exchange 安全清空" << endl;
  	} else {
    	cout << " 所有权未交出，内部指针未清空" << endl;
  	}

  	cout << "\n----------- 测试结束 -----------" << endl;
  	return 0;
}