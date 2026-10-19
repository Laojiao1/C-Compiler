#include "02_my_shared_ptr.hpp"
#include <iostream>
#include <string>

using namespace std;

class Tensor {
public:
    string name_;

    Tensor(string name) : name_(name) {
    	cout << "[Tensor] " << name_ << " 被创建了，分配物理内存！\n";
    }

    ~Tensor() {
    	cout << "[Tensor] " << name_ << " 被彻底销毁了，释放物理内存并关灯！\n";
    }
};

int main() {
  	cout << "----- 阶段 1：创建数据 -----" << endl;
  	ai_infra::shared_ptr<Tensor> p1(new Tensor("Shared_Node"));
  	cout << ">>> 当前引用人数: " << p1.use_count() << " (预期为 1)" << endl;

  	cout << "----- 阶段 2：创建新引用（触发拷贝构造）-----" << endl;
  	ai_infra::shared_ptr<Tensor> p2 = p1;
  	cout << ">>> p2 进场后，当前引用人数: " << p1.use_count() << " (预期为 2)" << endl;

  	cout << "----- 阶段 3：局部作用域（部分人离场）-----" << endl;
  	{
    	ai_infra::shared_ptr<Tensor> p3;
    	p3 = p1; // 触发拷贝赋值
    	cout << ">>> p3 进场后，当前引用人数: " << p1.use_count() << " (预期为 3)" << endl;
    	cout << ">>> p3 准备离开局部作用域..." << endl;
  	} // p3 在这里死亡，调用析构函数，内部控制块计数必须减 1

  	cout << ">>> p3 已经死亡！当前引用人数: " << p1.use_count() << " (预期为 2)" << endl;
  	cout << ">>> (注意：此时绝对不能打印出 Tensor 被销毁的日志，因为 p1 和 p2 还在！)" << endl;

  	cout << "----- 阶段 4：强行断开引用（测试赋值重分配） -----" << endl;
  	ai_infra::shared_ptr<Tensor> empty_ptr;
  	p1 = empty_ptr; // p1 强行断开旧连接，变成空壳
  	cout << ">>> p1 被强行踢出，此时仅剩 p2 还在里面。" << endl;
  	cout << ">>> p2 眼里的引用人数: " << p2.use_count() << " (预期为 1)" << endl;

  	cout << "----- 阶段 5：主函数结束，最后一个人准备离开 -----" << endl;
  	return 0;
}
// p2 在这里死亡，计数终于归 0，触发双重 delete（释放张量 + 释放控制块）