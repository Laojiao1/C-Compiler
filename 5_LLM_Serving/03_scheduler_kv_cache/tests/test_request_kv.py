"""验证按需扩容、容量不足时不部分分配，以及请求间块归属。"""

import unittest

from paged_kv import BlockPool, RequestKV, locate_token


class RequestKVTests(unittest.TestCase):
    def test_zero_target_uses_no_blocks(self):
        pool = BlockPool(3)
        request = RequestKV(pool, block_size=4)
        request.ensure_capacity(0)
        self.assertEqual(request.block_table, [])
        self.assertEqual(list(pool.free_blocks), [0, 1, 2])

    def test_growth_at_block_boundaries(self):
        pool = BlockPool(3)
        request = RequestKV(pool, block_size=4)
        for target, expected in [
            (1, [0]),
            (4, [0]),
            (5, [0, 1]),
            (8, [0, 1]),
            (9, [0, 1, 2]),
            (12, [0, 1, 2]),
        ]:
            with self.subTest(target=target):
                request.ensure_capacity(target)
                self.assertEqual(request.block_table, expected)

    def test_smaller_target_does_not_shrink(self):
        pool = BlockPool(3)
        request = RequestKV(pool, block_size=4)
        request.ensure_capacity(9)
        request.ensure_capacity(2)
        request.ensure_capacity(0)
        self.assertEqual(request.block_table, [0, 1, 2])
        self.assertEqual(pool.ref_counts, [1, 1, 1])

    def test_insufficient_blocks_preserve_state(self):
        pool = BlockPool(3)
        request = RequestKV(pool, block_size=4)
        request.ensure_capacity(4)
        with self.assertRaises(RuntimeError):
            request.ensure_capacity(16)
        self.assertEqual(request.block_table, [0])
        self.assertEqual(list(pool.free_blocks), [1, 2])
        self.assertEqual(pool.ref_counts, [1, 0, 0])
        request.ensure_capacity(12)
        self.assertEqual(request.block_table, [0, 1, 2])

    def test_release_all_is_repeatable_and_blocks_are_reusable(self):
        pool = BlockPool(2)
        request = RequestKV(pool, block_size=4)
        request.ensure_capacity(8)
        request.release_all()
        request.release_all()
        self.assertEqual(request.block_table, [])
        self.assertEqual(pool.ref_counts, [0, 0])
        self.assertEqual(list(pool.free_blocks), [0, 1])
        other = RequestKV(pool, block_size=4)
        other.ensure_capacity(8)
        self.assertEqual(other.block_table, [0, 1])
        self.assertEqual(pool.ref_counts, [1, 1])

    def test_interleaved_requests_have_separate_blocks(self):
        pool = BlockPool(4)
        request_a = RequestKV(pool, block_size=4)
        request_b = RequestKV(pool, block_size=4)
        request_a.ensure_capacity(4)
        request_b.ensure_capacity(4)
        request_a.ensure_capacity(8)
        self.assertEqual(request_a.block_table, [0, 2])
        self.assertEqual(request_b.block_table, [1])
        self.assertEqual(locate_token(request_a.block_table, 4, 4), (2, 0))
        request_a.release_all()
        self.assertEqual(request_b.block_table, [1])
        self.assertEqual(pool.ref_counts, [0, 1, 0, 0])

    def test_negative_target_preserves_state(self):
        pool = BlockPool(2)
        request = RequestKV(pool, block_size=4)
        with self.assertRaises(ValueError):
            request.ensure_capacity(-1)
        self.assertEqual(request.block_table, [])
        self.assertEqual(list(pool.free_blocks), [0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
