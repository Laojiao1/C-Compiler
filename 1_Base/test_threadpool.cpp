#include "03_my_threadpool.hpp"
#include <chrono>
#include <iostream>
#include <ostream>
#include <sstream>
#include <thread>
#include <vector>

using namespace std;
using namespace ai_infra;

// 模拟一个耗时的算子计算（比如矩阵乘法）
string simulate_matrix_mul(int task_id) {
    // 获取当前干活的打工人（线程）ID
    stringstream ss;
    ss << this_thread::get_id();
    string thread_id = ss.str();

    // 模拟计算耗时 (休眠 100 毫秒)
    this_thread::sleep_for(chrono::milliseconds(100));

    return "任务 [" + to_string(task_id) + "] 计算完成！(由打工人 " + thread_id +" 执行)" ;
}

int main() {
    cout << "------ 引擎启动：初始化具有 4 个打工人的线程池 ------" << endl;
    ThreadPool pool(4);

    // 用来存放任务执行结果的“取件回执单” (std::future)
    vector<future<string>> results;

    cout << "------ 派发任务：向队列狂塞 20 个独立的任务 ------" << endl;
    for (int i = 0; i < 20; ++i) {
        // pool.enqueue 会立刻返回一个 future，主线程不会在这里卡住
        results.emplace_back(pool.enqueue(simulate_matrix_mul, i));
    }

    cout << ">>> 主线程派发完毕！主线程挂起，后台 4 个打工人正在抢单..." << endl;

    // 阻塞等待，回收所有任务的计算结果
    for (size_t i = 0; i < results.size(); ++i) {
        // future.get() 会一直阻塞，直到对应的那个任务被执行完并 return 出结果
        cout << results[i].get() << endl;
    }

    cout << "------ 所有任务执行完毕，触发池的析构函数 ------" << endl;

    return 0;
}
