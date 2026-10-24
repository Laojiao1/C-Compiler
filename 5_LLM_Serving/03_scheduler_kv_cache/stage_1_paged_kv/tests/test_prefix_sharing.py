"""验证完整块共享后的引用生命周期、读取与分支追加。"""

import unittest

import torch

from stage_1_paged_kv.paged_kv import BlockPool, RequestKV
from stage_1_paged_kv.paged_kv_store import PagedKVStore


class PrefixSharingTests(unittest.TestCase):
    def make_source(self):
        pool = BlockPool(5)
        store = PagedKVStore(pool, block_size=4, num_kv_heads=2, head_dim=3)
        store.k_pool.fill_(float("nan"))
        store.v_pool.fill_(float("nan"))
        source = RequestKV(pool, block_size=4)
        k = torch.arange(60, dtype=torch.float32).reshape(10, 2, 3)
        v = k + 1000
        store.append(source, k, v)
        return pool, store, source, k, v

    def assert_kv_equal(self, actual, expected):
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def test_retain_keeps_block_live_until_last_release(self):
        pool = BlockPool(2)
        block = pool.allocate()
        pool.retain(block)
        self.assertEqual(pool.ref_counts, [2, 0])
        self.assertEqual(list(pool.free_blocks), [1])
        pool.release(block)
        self.assertEqual(pool.ref_counts, [1, 0])
        self.assertNotIn(block, pool.free_blocks)
        pool.release(block)
        self.assertEqual(pool.ref_counts, [0, 0])
        self.assertEqual(list(pool.free_blocks), [1, 0])

    def test_retain_rejects_invalid_or_free_blocks(self):
        pool = BlockPool(2)
        pool.allocate()
        for block_id in (-1, 2):
            with self.subTest(block_id=block_id):
                with self.assertRaises(IndexError):
                    pool.retain(block_id)
        with self.assertRaises(ValueError):
            pool.retain(1)
        self.assertEqual(pool.ref_counts, [1, 0])
        self.assertEqual(list(pool.free_blocks), [1])

    def test_sharing_uses_existing_storage_and_separate_block_tables(self):
        pool, store, source, k, v = self.make_source()
        branch = source.share_prefix(8)
        self.assertIs(branch.pool, pool)
        self.assertEqual(branch.block_size, 4)
        self.assertIsNot(branch.block_table, source.block_table)
        self.assertEqual(source.block_table, [0, 1, 2])
        self.assertEqual(branch.block_table, [0, 1])
        self.assertEqual(source.num_tokens, 10)
        self.assertEqual(branch.num_tokens, 8)
        self.assertEqual(pool.ref_counts, [2, 2, 1, 0, 0])
        self.assertEqual(list(pool.free_blocks), [3, 4])
        self.assert_kv_equal(store.gather(branch), (k[:8], v[:8]))

    def test_branches_append_to_private_tails(self):
        pool, store, source, k, v = self.make_source()
        branch = source.share_prefix(8)
        branch_k = torch.full((1, 2, 3), 300.0)
        branch_v = branch_k + 1000
        source_k = torch.full((1, 2, 3), 500.0)
        source_v = source_k + 1000
        store.append(branch, branch_k, branch_v)
        store.append(source, source_k, source_v)
        self.assertEqual(source.block_table, [0, 1, 2])
        self.assertEqual(branch.block_table, [0, 1, 3])
        self.assertEqual(pool.ref_counts, [2, 2, 1, 1, 0])
        self.assert_kv_equal(
            store.gather(source),
            (torch.cat([k, source_k]), torch.cat([v, source_v])),
        )
        self.assert_kv_equal(
            store.gather(branch),
            (torch.cat([k[:8], branch_k]), torch.cat([v[:8], branch_v])),
        )

    def test_releasing_either_request_preserves_survivor(self):
        for first_to_finish in ("source", "branch"):
            with self.subTest(first_to_finish=first_to_finish):
                pool, store, source, k, v = self.make_source()
                branch = source.share_prefix(8)
                if first_to_finish == "source":
                    source.release_all()
                    survivor, expected = branch, (k[:8], v[:8])
                else:
                    branch.release_all()
                    survivor, expected = source, (k, v)
                self.assertEqual(pool.ref_counts[:2], [1, 1])
                self.assertNotIn(0, pool.free_blocks)
                self.assertNotIn(1, pool.free_blocks)
                self.assert_kv_equal(store.gather(survivor), expected)
                survivor.release_all()
                self.assertEqual(pool.ref_counts, [0, 0, 0, 0, 0])
                self.assertEqual(sorted(pool.free_blocks), list(range(5)))

    def test_invalid_prefix_preserves_state(self):
        pool, _, source, _, _ = self.make_source()
        # 容量可以预留到 16，但有效历史仍只有 10，不能共享第 11、12 个位置。
        source.ensure_capacity(16)
        for prefix_tokens in (-4, 3, 10, 12):
            with self.subTest(prefix_tokens=prefix_tokens):
                with self.assertRaises(ValueError):
                    source.share_prefix(prefix_tokens)
                self.assertEqual(source.num_tokens, 10)
                self.assertEqual(source.block_table, [0, 1, 2, 3])
                self.assertEqual(pool.ref_counts, [1, 1, 1, 1, 0])
                self.assertEqual(list(pool.free_blocks), [4])

    def test_zero_prefix_creates_empty_request_without_references(self):
        pool, store, source, _, _ = self.make_source()
        branch = source.share_prefix(0)
        self.assertEqual(branch.block_table, [])
        self.assertEqual(branch.num_tokens, 0)
        self.assertEqual(pool.ref_counts, [1, 1, 1, 0, 0])
        self.assertEqual(list(pool.free_blocks), [3, 4])
        k, v = store.gather(branch)
        self.assertEqual(tuple(k.shape), (0, 2, 3))
        self.assertEqual(tuple(v.shape), (0, 2, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
