"""验证按最近使用顺序淘汰、活跃块保护，以及释放容量后的再次分配。"""

import unittest

import torch

from paged_kv import BlockPool, RequestKV
from paged_kv_store import PagedKVStore
from prefix_cache import PrefixCache


class PrefixEvictionTests(unittest.TestCase):
    def make_idle_entries(self, num_entries=2):
        pool = BlockPool(num_entries)
        store = PagedKVStore(pool, 4, 1, 2)
        cache = PrefixCache(pool, 4, "demo:fp32")
        token_lists = []
        for index in range(num_entries):
            # 使用不同的完整首块，避免一个前缀缺块影响另一个前缀的命中。
            tokens = list(range(index * 10, index * 10 + 4))
            request = RequestKV(pool, 4)
            k = torch.full((4, 1, 2), float(index + 1))
            store.append(request, k, k + 100)
            cache.register(request, tokens)
            request.release_all()
            token_lists.append(tokens)
        return pool, store, cache, token_lists

    def test_empty_cache_has_nothing_to_evict(self):
        pool = BlockPool(2)
        cache = PrefixCache(pool, 4, "demo:fp32")
        self.assertIsNone(cache.evict_one())
        self.assertEqual(list(cache.entries.items()), [])
        self.assertEqual(pool.ref_counts, [0, 0])
        self.assertEqual(list(pool.free_blocks), [0, 1])

    def test_eviction_restores_capacity_and_removes_stale_key(self):
        pool, store, cache, tokens = self.make_idle_entries(num_entries=1)
        request = RequestKV(pool, 4)
        # 唯一的块还被缓存引用占用，所以直接扩容会失败。
        with self.assertRaises(RuntimeError):
            request.ensure_capacity(4)
        self.assertEqual(cache.evict_one(), 0)
        self.assertEqual(list(cache.entries.items()), [])
        self.assertEqual(pool.ref_counts, [0])
        self.assertEqual(list(pool.free_blocks), [0])
        request.ensure_capacity(4)
        self.assertEqual(request.block_table, [0])
        new_k = torch.full((4, 1, 2), 42.0)
        store.append(request, new_k, new_k + 100)
        torch.testing.assert_close(store.gather(request)[0], new_k, rtol=0, atol=0)
        # 物理块已被新内容覆盖；旧指纹必须失效，不能再把它交给旧输入。
        miss = cache.acquire(tokens[0])
        self.assertEqual(miss.block_table, [])
        self.assertEqual(miss.num_tokens, 0)
        self.assertEqual(pool.ref_counts, [1])

    def test_recent_hit_changes_which_idle_entry_is_evicted(self):
        pool, _, cache, tokens = self.make_idle_entries()
        first_hash, second_hash = list(cache.entries)
        request = cache.acquire(tokens[0])
        request.release_all()
        self.assertEqual(list(cache.entries), [second_hash, first_hash])
        self.assertEqual(cache.evict_one(), 1)
        self.assertEqual(list(cache.entries), [first_hash])
        self.assertEqual(pool.ref_counts, [1, 0])
        self.assertEqual(list(pool.free_blocks), [1])
        self.assertEqual(cache.acquire(tokens[1]).num_tokens, 0)
        survivor = cache.acquire(tokens[0])
        self.assertEqual(survivor.block_table, [0])

    def test_oldest_active_entry_is_skipped(self):
        pool, store, cache, tokens = self.make_idle_entries()
        request_a = cache.acquire(tokens[0])
        request_b = cache.acquire(tokens[1])
        request_b.release_all()
        # 顺序为 A、B；A 更旧但仍活跃，只能淘汰较新的空闲 B。
        self.assertEqual(list(cache.entries.values()), [0, 1])
        self.assertEqual(pool.ref_counts, [2, 1])
        self.assertEqual(cache.evict_one(), 1)
        self.assertEqual(list(cache.entries.values()), [0])
        self.assertEqual(pool.ref_counts, [2, 0])
        expected_k = torch.ones(4, 1, 2)
        actual_k, actual_v = store.gather(request_a)
        torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(actual_v, expected_k + 100, rtol=0, atol=0)

    def test_all_active_entries_preserve_state_until_request_releases(self):
        pool, store, cache, tokens = self.make_idle_entries()
        request_a = cache.acquire(tokens[0])
        request_b = cache.acquire(tokens[1])
        original_order = list(cache.entries.items())
        self.assertIsNone(cache.evict_one())
        self.assertEqual(list(cache.entries.items()), original_order)
        self.assertEqual(pool.ref_counts, [2, 2])
        self.assertEqual(list(pool.free_blocks), [])
        request_a.release_all()
        self.assertEqual(cache.evict_one(), 0)
        self.assertEqual(pool.ref_counts, [0, 2])
        expected_k = torch.full((4, 1, 2), 2.0)
        torch.testing.assert_close(store.gather(request_b)[0], expected_k, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
