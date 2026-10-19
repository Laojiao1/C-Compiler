#pragma once

#include <type_traits>
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

namespace ai_infra {
    
class ThreadPool {
public:
    // 构造函数，用于启动一定数量的线程
    ThreadPool(size_t threads);
    
    // 生产者：生产任务
    template<class F, class... Args> 
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    // 析构函数
    ~ThreadPool();
    
private:
    // 线程集合
    std::vector<std::thread> workers;

    // 任务集合（统一封装成了无参数，无返回值的格式，便于执行）
    std::queue<std::function<void()>> tasks;

    // 互斥锁，防止几个工人同时抢一个任务
    std::mutex queue_mutex;

    // 条件变量
    std::condition_variable condition;

    // 关闭生产工厂的标志，默认为 false
    bool stop;
};

// 构造函数，ThreadPool::TreadPool 用于指明这是ThreadPool类的构造函数
inline ThreadPool::ThreadPool(size_t threads) : stop(false) {
    // 遍历线程数量
    for(size_t i = 0; i < threads; i++) {
        // 采用 emplace_back，功能与 push_back 一样，但是更高效
        workers.emplace_back(
            [this]()-> void { // 采用 C++11 的 Lambda表达式(匿名函数)，它可以直接使用当前类(ThreadPool)的成员变量。这里的功能是传递给thread一个匿名函数，相当于告诉线程：一出生，就执行这段代码
                
                while(1) { // 陷入死循环
                    std::function<void()> task;
                    // 添加一个局部作用域{}，用于控制锁的寿命
                    {
                        // 加锁
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        
                        // 条件检查：满足条件（有活干），去干活，上锁
                        // 不满足条件（没活干），挂起，释放锁
                        // condition.wait 会在条件不满足时阻塞，并释放锁，此时不占用CPU资源
                        // 当被 notify 唤醒时，它会重获锁，再次检查条件
                        this->condition.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                        });
                        
                        // 如果工厂倒闭或者没任务了，就结束线程
                        if(this->stop && this->tasks.empty()) 
                            return;

                        // 拿任务去干活
                        // 注意这里用 move “偷”过来，节省资源，省去不必要的资源拷贝
                        task = std::move(this->tasks.front());

                        // 记得出队
                        this->tasks.pop();

                    } // 注意：离开这个{}作用域，局部变量 lock 自动销毁，把锁自动让给别人
                    
                    // 开始干活，注意一定要放在{}作用域外面，否则线程会被卡死！
                    task();
                }
            }
        );
    }
}

// 生产者逻辑：
template<class F, class... Args>
    auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type>
    
{
    // 相当于 typedef
    using return_type = typename std::result_of<F(Args...)>::type;
    
    // 这块是为了把带参数和返回值的任意函数，统一打包成 void() 的格式
    // std::bind 将函数 f 和它所需要的参数 args 绑定在一起，变成一个没有参数的函数
    // std::packaged_task<return_type()> 捕获返回值
    // 结果：带返回值的函数 -> 变成了返回值为 void 但会往里塞结果的函数
    // packaged_task 不允许被复制，只能被移动，但我们需要某些任务可以被复制，所以需要用智能指针包裹起来，让它能够在多线程之间流转，推入队列
    auto task = std::make_shared< std::packaged_task<return_type()> > (
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> res = task->get_future(); // 刚才使用 std::packaged_task 返回的是一个集成了“函数执行外壳”和“异步数据管道入口”的复合对象。通过 get_future() 收货

    {
        // 在生成任务时，需要加锁
        std::unique_lock<std::mutex> lock(queue_mutex);

        // 如果生产工厂停止运作了，就拒绝，报错
        if(this->stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        // 把任务放入队列中
        // 这里采用 emplace，和 push 不一样的是，emplace 直接在队列中构造一个函数对象，避免了不必要的拷贝
        this->tasks.emplace([task](){ (*task)(); });

    } // 离开{}，自动解锁
    
    // 唤醒随机一个线程
    condition.notify_one();
    
    // 返回给了调用 enqueue 函数的那个线程
    return res;
}
    
// 析构函数
inline ThreadPool::~ThreadPool() {
    {
        // 加锁
        std::unique_lock<std::mutex> lock(queue_mutex);

        // 工厂关闭标志    
        stop = true;
    }

    // 唤醒所有阻塞的线程，处理遗留任务或直接销毁
    condition.notify_all();
    
    // 确认所有线程都已经死亡
    // join()的意思是：等待该线程执行完毕，并且在它彻底结束之前，调用它的主线程必须原地停下，不能往后执行
    for(std::thread &worker: workers) {
        if(worker.joinable()) {
            worker.join(); 
        }
    }
};   

} 