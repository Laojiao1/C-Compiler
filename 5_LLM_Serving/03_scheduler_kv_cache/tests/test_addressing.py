"""只验证地址转换，不验证 K/V 数据有效性或分配策略。"""

import unittest

from paged_kv import locate_token


class AddressingTests(unittest.TestCase):
    def test_noncontiguous_blocks_and_boundaries(self):
        table = [5, 1, 8]
        cases = [
            (0, (5, 0)),
            (2, (5, 2)),
            (3, (5, 3)),
            (4, (1, 0)),
            (6, (1, 2)),
            (7, (1, 3)),
            (8, (8, 0)),
            (9, (8, 1)),
        ]
        for position, expected in cases:
            with self.subTest(position=position):
                self.assertEqual(locate_token(table, 4, position), expected)

    def test_block_size_one(self):
        self.assertEqual(locate_token([9, 2], 1, 1), (2, 0))

    def test_invalid_parameters(self):
        for block_size, position in [(0, 0), (-4, 0), (4, -1)]:
            with self.subTest(block_size=block_size, position=position):
                with self.assertRaises(ValueError):
                    locate_token([5, 1, 8], block_size, position)

    def test_missing_mapping(self):
        for table, position in [([5, 1, 8], 12), ([], 0)]:
            with self.subTest(table=table, position=position):
                with self.assertRaises(IndexError):
                    locate_token(table, 4, position)


if __name__ == "__main__":
    unittest.main(verbosity=2)
