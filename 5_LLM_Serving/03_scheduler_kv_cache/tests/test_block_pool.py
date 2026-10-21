"""验证块号分配、容量不足及回收复用，不涉及 GPU 分配或性能。"""

import unittest

from paged_kv import BlockPool


class BlockPoolTests(unittest.TestCase):
    def test_unique_allocations(self):
        pool = BlockPool(3)
        self.assertEqual([pool.allocate() for _ in range(3)], [0, 1, 2])
        self.assertEqual(pool.ref_counts, [1, 1, 1])
        self.assertEqual(list(pool.free_blocks), [])

    def test_exhaustion_preserves_state(self):
        pool = BlockPool(1)
        pool.allocate()
        with self.assertRaises(RuntimeError):
            pool.allocate()
        self.assertEqual(pool.ref_counts, [1])
        self.assertEqual(list(pool.free_blocks), [])

    def test_reuse_released_block(self):
        pool = BlockPool(3)
        blocks = [pool.allocate() for _ in range(3)]
        pool.release(blocks[1])
        self.assertEqual(pool.ref_counts, [1, 0, 1])
        self.assertEqual(list(pool.free_blocks), [1])
        self.assertEqual(pool.allocate(), blocks[1])
        self.assertEqual(pool.ref_counts, [1, 1, 1])

    def test_double_release_is_rejected(self):
        pool = BlockPool(1)
        block = pool.allocate()
        pool.release(block)
        with self.assertRaises(ValueError):
            pool.release(block)
        self.assertEqual(list(pool.free_blocks), [block])
        self.assertEqual(pool.ref_counts, [0])
        self.assertEqual(pool.allocate(), block)
        with self.assertRaises(RuntimeError):
            pool.allocate()

    def test_invalid_release_preserves_state(self):
        pool = BlockPool(2)
        pool.allocate()
        for invalid_id in (-1, 2):
            with self.subTest(block_id=invalid_id):
                with self.assertRaises(IndexError):
                    pool.release(invalid_id)
        with self.assertRaises(ValueError):
            pool.release(1)
        self.assertEqual(pool.ref_counts, [1, 0])
        self.assertEqual(list(pool.free_blocks), [1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
