"""验证指纹到物理块的关联、缓存持有引用及新请求领取。"""

import unittest

import torch

from stage_1_paged_kv.paged_kv import BlockPool, RequestKV
from stage_1_paged_kv.paged_kv_store import PagedKVStore
from stage_1_paged_kv.prefix_cache import PrefixCache, build_prefix_hashes


class PrefixCacheTests(unittest.TestCase):
    def make_source(self):
        pool = BlockPool(6)
        store = PagedKVStore(pool, 4, 2, 3)
        store.k_pool.fill_(float("nan"))
        store.v_pool.fill_(float("nan"))
        source = RequestKV(pool, 4)
        tokens = list(range(10))
        k = torch.arange(60, dtype=torch.float32).reshape(10, 2, 3)
        v = k + 1000
        store.append(source, k, v)
        cache = PrefixCache(pool, 4, "demo:revision-1:fp32")
        return pool, store, source, tokens, k, v, cache

    def assert_kv_equal(self, actual, expected):
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def test_register_maps_only_complete_blocks_and_pins_them(self):
        pool, _, source, tokens, _, _, cache = self.make_source()
        cache.register(source, tokens)
        hashes = build_prefix_hashes(tokens, 4, cache.namespace)
        self.assertEqual(list(cache.entries.items()), [(hashes[0], 0), (hashes[1], 1)])
        self.assertEqual(pool.ref_counts, [2, 2, 1, 0, 0, 0])
        self.assertEqual(list(pool.free_blocks), [3, 4, 5])
        self.assertEqual(source.num_tokens, 10)

    def test_cache_outlives_source_and_new_request_appends_privately(self):
        pool, store, source, tokens, k, v, cache = self.make_source()
        cache.register(source, tokens)
        source.release_all()
        self.assertEqual(pool.ref_counts, [1, 1, 0, 0, 0, 0])
        self.assertEqual(list(pool.free_blocks), [3, 4, 5, 2])
        request = cache.acquire(list(range(8)) + [90, 91])
        self.assertEqual(request.block_table, [0, 1])
        self.assertEqual(request.num_tokens, 8)
        self.assertEqual(pool.ref_counts, [2, 2, 0, 0, 0, 0])
        self.assert_kv_equal(store.gather(request), (k[:8], v[:8]))
        new_k = torch.full((2, 2, 3), 300.0)
        new_v = new_k + 1000
        store.append(request, new_k, new_v)
        self.assertEqual(request.block_table, [0, 1, 3])
        self.assert_kv_equal(
            store.gather(request),
            (torch.cat([k[:8], new_k]), torch.cat([v[:8], new_v])),
        )
        request.release_all()
        self.assertEqual(pool.ref_counts, [1, 1, 0, 0, 0, 0])
        self.assertNotIn(0, pool.free_blocks)
        self.assertNotIn(1, pool.free_blocks)

    def test_duplicate_registration_does_not_leak_cache_references(self):
        pool, store, source, tokens, k, v, cache = self.make_source()
        cache.register(source, tokens)
        cache.register(source, tokens)
        self.assertEqual(pool.ref_counts, [2, 2, 1, 0, 0, 0])
        # 同样内容也可能已经在另一组物理块上重复算过，索引仍保留最早登记的块。
        duplicate = RequestKV(pool, 4)
        store.append(duplicate, k, v)
        self.assertEqual(duplicate.block_table, [3, 4, 5])
        cache.register(duplicate, tokens)
        self.assertEqual(list(cache.entries.values()), [0, 1])
        self.assertEqual(pool.ref_counts, [2, 2, 1, 1, 1, 1])
        duplicate.release_all()
        self.assertEqual(pool.ref_counts, [2, 2, 1, 0, 0, 0])

    def test_lookup_returns_contiguous_prefix_and_updates_recency(self):
        pool, store, source, tokens, k, v, cache = self.make_source()
        cache.register(source, tokens)
        hashes = list(cache.entries)
        request = cache.acquire(list(range(4)) + [90, 91, 92, 93])
        self.assertEqual(request.num_tokens, 4)
        self.assertEqual(request.block_table, [0])
        self.assert_kv_equal(store.gather(request), (k[:4], v[:4]))
        self.assertEqual(list(cache.entries), [hashes[1], hashes[0]])
        before = pool.ref_counts.copy()
        miss = cache.acquire([99] + list(range(1, 8)))
        self.assertEqual(miss.num_tokens, 0)
        self.assertEqual(miss.block_table, [])
        self.assertEqual(pool.ref_counts, before)

    def test_lookup_stops_at_missing_first_block(self):
        pool, _, source, tokens, _, _, cache = self.make_source()
        cache.register(source, tokens)
        first_hash = next(iter(cache.entries))
        # 模拟后续将实现的淘汰：去掉首块索引并释放缓存引用，后块索引仍在。
        removed_block = cache.entries.pop(first_hash)
        pool.release(removed_block)
        before = pool.ref_counts.copy()
        request = cache.acquire(tokens)
        self.assertEqual(request.num_tokens, 0)
        self.assertEqual(request.block_table, [])
        self.assertEqual(pool.ref_counts, before)

    def test_empty_and_partial_inputs_do_not_allocate_or_cache_blocks(self):
        pool = BlockPool(2)
        cache = PrefixCache(pool, 4, "demo:fp32")
        cache.register(RequestKV(pool, 4), [])
        for tokens in ([], [1, 2, 3]):
            request = cache.acquire(tokens)
            self.assertEqual(request.num_tokens, 0)
            self.assertEqual(request.block_table, [])
        self.assertEqual(list(cache.entries), [])
        self.assertEqual(pool.ref_counts, [0, 0])
        self.assertEqual(list(pool.free_blocks), [0, 1])

    def test_invalid_registration_preserves_state(self):
        pool, _, source, tokens, _, _, cache = self.make_source()
        cases = [
            (source, tokens[:-1]),
            (source, [-1] + tokens[1:]),
            (RequestKV(BlockPool(6), 4), []),
            (RequestKV(pool, 2), []),
        ]
        for request, ids in cases:
            with self.subTest(length=len(ids), block_size=request.block_size):
                with self.assertRaises(ValueError):
                    cache.register(request, ids)
        self.assertEqual(list(cache.entries), [])
        self.assertEqual(pool.ref_counts, [1, 1, 1, 0, 0, 0])
        self.assertEqual(list(pool.free_blocks), [3, 4, 5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
