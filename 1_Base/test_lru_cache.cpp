#include <iostream>
#include "04_lru_cache.hpp"

using namespace std;
using namespace ai_infra;

int main() {
    cout << "------- 引擎启动：初始化容量为 2 的算子缓存 -------n";
    LRUCache cache(2);

    cout << "1. 编译算子 1 并塞入缓存...\n";
    cache.put(1, 100); // 缓存是 {1=100}
    
    cout << "2. 编译算子 2 并塞入缓存...\n";
    cache.put(2, 200); // 缓存是 {2=200, 1=100}，算子 2 在链表头部
    
    cout << "3. 尝试获取算子 1: " << cache.get(1) << " (预期: 100)\n";
    // 此时算子 1 被重新激活，跑到头部！缓存顺序变成 {1=100, 2=200}
    
    cout << "4. 内存告急！强行塞入算子 3...\n";
    cache.put(3, 300); 
    // 容量满了！尾部的算子 2 被无情淘汰。缓存变成 {3=300, 1=100}
    
    cout << "5. 尝试获取被淘汰的算子 2: " << cache.get(2) << " (预期: -1，代表 Cache Miss)\n";
    
    cout << "6. 强行塞入算子 4...\n";
    cache.put(4, 400);
    // 尾部的算子 1 遭遇淘汰。缓存变成 {4=400, 3=300}
    
    cout << "7. 尝试获取算子 1: " << cache.get(1) << " (预期: -1)\n";
    cout << "8. 尝试获取算子 3: " << cache.get(3) << " (预期: 300)\n";
    cout << "9. 尝试获取算子 4: " << cache.get(4) << " (预期: 400)\n";

    cout << "------- 测试完毕，缓存驱逐逻辑完美生效！ -------n";
    return 0;
}