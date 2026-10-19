#pragma once

#include <unordered_map>
#include <list>
#include <utility>

namespace ai_infra {
    
class LRUCache {
private:
    // 缓存容量上限
    int capacity_;
    
    // 双向链表：用于维护数据访问的顺序，存放对组，以后可以通过对组中的Key去哈希表里直接对其进行操作
    std::list<std::pair<int, int>> cache_list_;
    
    // 哈希表：用于实现O(1)级别的快速查找
    // Key 值存放双向链表中对应节点的迭代器，这样拿到迭代器之后就可以在O(1)d的时间内去直接操作链表中的节点
    std::unordered_map< int, 
    	std::list<std::pair<int, int>>::iterator > cache_map_;

public:
    // 构造函数，explicit 关键字防止隐式转换
	explicit LRUCache(int capacity) : capacity_(capacity) {}
    
    // ------- 读取缓存 -------
    int get(int key) {
        // 在哈希表中查找key，看是否能命中
        auto it = cache_map_.find(key);
        // 如果没命中，返回-1
        if(it == cache_map_.end()) {
            return -1;
        }
        // 如果命中了，需要从哈希表中拿出对应节点的迭代器
        auto node_it = it->second;
        
        // 然后将该节点移动到链表的最头部，代表它刚刚被访问过
        // 这里采用std::splice,它可以通过改变指针指向，在不破坏迭代器顺序的前提下，实现一系列操作 
        // std::splice(目标位置，源链表，想要移动的节点)
        cache_list_.splice(cache_list_.begin(), cache_list_, node_it);
        // 最后返回节点的 Value
        return node_it->second;
    }
    
    // ------- 往队列中写入或者更新队列缓存 -------
    void put(int key, int value) {
        // 先查找一下哈希表，看看这个 key 是否已经存在
        auto it = cache_map_.find(key);
        
        // 如果存在的话，更新value，然后把该节点移动到头部即可
        if(it != cache_map_.end()) {
            auto node_it = it->second;
            node_it->second = value; // 更新value
            cache_list_.splice(cache_list_.begin(), cache_list_, node_it); // 移动到头部
            return;
        }
        
        // 如果不存在，说明是新数据，需要插入
        // 这里仍然采用 emplace_front，它是 push_front 的升级版，直接在链表头部分配好的内存块上，调用 pair 的构造函数原地生成对象。无需像 push_front 还得在外面打包成 pair，才拷贝到链表中
        cache_list_.emplace_front(key, value);
        // 在哈希表中添加一条记录，指向刚插入的头部节点
        cache_map_[key] = cache_list_.begin();
        
        // 最后还需要检查一下当前缓存数量是否超出了上限
        // 如果超出上限，需要进行删除
        if(cache_map_.size() > capacity_) {
            // 拿到尾部数据（最久未使用）
            auto last_pair = cache_list_.back();
            // 删除哈希表里对应的记录
            cache_map_.erase(last_pair.first);
            // 记得把链表中的节点也删除
            cache_list_.pop_back();
        }
    }
};
    
} // namespace ai_infra