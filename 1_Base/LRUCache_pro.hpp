// 力扣 147 题
// 最终战绩：
// - 执行用时分布：37 ms 击败 99.86%；
// - 消耗内存分布：168.78MB 击败 95.59%

#include <bits/stdc++.h>
using namespace std;

// 1. 关闭 IO 同步，极限加速读写
static const auto _ = []() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

class LRUCache {
private:
    // 双向链表
    struct Node {
        int key, value;
        Node* prev;
        Node* next;
    };

    int capacity_; // 容量
    int size_; // 当前缓存数量

    // 优化 3：创建一个不存任何真实数据的循环空节点 dummy
    // dummy->next 永远都是最新的节点，dummy->prev 永远都是最旧的节点
    Node* dummy;          

    Node* hash_map[10005];  // 优化 4：直接用数组模拟哈希表，避免 unordered_map 的哈希冲突和内存碎片化问题
    Node* node_pool;        // 优化 1：直接申请一块连续的内存池，不现场 new

    // 
    // 辅助函数：将节点插入到头部（dummy 的后面）
    inline void addToHead(Node* node) {
        node->next = dummy->next;
        node->prev = dummy;
        dummy->next->prev = node;
        dummy->next = node;
    }

    // 辅助函数：将已存在的节点移动到头部
    inline void moveToHead(Node* node) {
        // 先从原位置拔出来
        node->prev->next = node->next;
        node->next->prev = node->prev;
        // 再插到头部
        addToHead(node);
    }

public:
    LRUCache(int capacity) : capacity_(capacity), size_(0) {
        // 优化 2：采用 memset，底层是指令级清空数组（采用块级别向量化指令），碾压 for 循环
        memset(hash_map, 0, sizeof(hash_map));

        // 优化 1：直接一次性向系统申请连续的内存空间
        node_pool = new Node[capacity_]; 

        // 优化 3：空节点闭环，自己连自己
        dummy = new Node();
        dummy->next = dummy;
        dummy->prev = dummy;
    }
    
    // LeetCode 会测试内存泄漏，虽然竞赛不管，但为了完美还是加上析构
    ~LRUCache() {
        delete[] node_pool;
        delete dummy;
    }
    
    int get(int key) {
        Node* node = hash_map[key];
        if (!node) return -1;
        moveToHead(node);
        return node->value;
    }
    
    void put(int key, int value) {
        Node* node = hash_map[key];
        if (node) { // 节点存在
            // 更新并前移
            node->value = value;
            moveToHead(node);
        } else {
            // 节点不存在，需要插入
            if (size_ == capacity_) { // 容量满，淘汰尾部（dummy 的前一个）
                Node* old_node = dummy->prev; 
                
                hash_map[old_node->key] = nullptr; 
                
                // 优化 5：直接复用旧节点，避免 new/delete
                old_node->key = key;     
                old_node->value = value;
                
                hash_map[key] = old_node; 
                moveToHead(old_node);     
            } else {
                // 容量未满：绝对不 new！直接从连续内存池里按顺序拿一个出来用
                Node* newNode = &node_pool[size_];
                newNode->key = key;
                newNode->value = value;
                
                hash_map[key] = newNode;
                addToHead(newNode);
                size_++;
            }
        }
    }
};