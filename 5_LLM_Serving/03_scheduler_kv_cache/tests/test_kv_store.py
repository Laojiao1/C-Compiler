"""用确定性小张量检查物理写入、跨块读取和有效长度，无需 GPU。"""

import unittest

import torch

from paged_kv import BlockPool, RequestKV
from paged_kv_store import PagedKVStore


def make_kv(n: int, start: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.arange(start, start + n * 6, dtype=torch.float32).reshape(n, 2, 3)
    return k, k + 1000


class KVStoreTests(unittest.TestCase):
    def make_store(self, num_blocks=4):
        pool = BlockPool(num_blocks)
        store = PagedKVStore(pool, block_size=4, num_kv_heads=2, head_dim=3)
        # 让未写入区域可辨认，避免尾槽错误被偶然的初值掩盖。
        store.k_pool.fill_(float("nan"))
        store.v_pool.fill_(float("nan"))
        return pool, store, RequestKV(pool, block_size=4)

    def assert_kv_equal(self, actual, expected):
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def test_chunk_roundtrip_and_physical_slots(self):
        _, store, request = self.make_store()
        kv = make_kv(6)
        store.append(request, *kv)
        self.assertEqual(request.num_tokens, 6)
        self.assertEqual(request.block_table, [0, 1])
        self.assert_kv_equal(store.gather(request), kv)
        for position, block_id, offset in [(0, 0, 0), (3, 0, 3), (4, 1, 0), (5, 1, 1)]:
            self.assert_kv_equal(
                (store.k_pool[block_id, offset], store.v_pool[block_id, offset]),
                (kv[0][position], kv[1][position]),
            )

    def test_incremental_append_preserves_history(self):
        _, store, request = self.make_store()
        first = make_kv(3)
        second = make_kv(2, start=200)
        store.append(request, *first)
        store.append(request, *second)
        self.assertEqual(request.num_tokens, 5)
        expected = tuple(torch.cat([a, b], dim=0) for a, b in zip(first, second))
        self.assert_kv_equal(store.gather(request), expected)

    def test_interleaved_requests_and_extra_reservation(self):
        pool, store, request_a = self.make_store()
        request_b = RequestKV(pool, block_size=4)
        first = make_kv(4)
        other = make_kv(4, start=100)
        extra = make_kv(1, start=200)
        store.append(request_a, *first)
        store.append(request_b, *other)
        store.append(request_a, *extra)
        self.assertEqual(request_a.block_table, [0, 2])
        self.assertEqual(request_b.block_table, [1])
        request_a.ensure_capacity(12)
        self.assertEqual(request_a.num_tokens, 5)
        expected_a = tuple(torch.cat([a, b], dim=0) for a, b in zip(first, extra))
        self.assert_kv_equal(store.gather(request_a), expected_a)
        self.assert_kv_equal(store.gather(request_b), other)

    def test_capacity_failure_preserves_valid_history(self):
        pool, store, request = self.make_store(num_blocks=2)
        first = make_kv(3)
        store.append(request, *first)
        with self.assertRaises(RuntimeError):
            store.append(request, *make_kv(6, start=200))
        self.assertEqual(request.num_tokens, 3)
        self.assertEqual(request.block_table, [0])
        self.assertEqual(list(pool.free_blocks), [1])
        self.assertEqual(pool.ref_counts, [1, 0])
        self.assert_kv_equal(store.gather(request), first)

    def test_empty_history_release_and_reuse(self):
        pool, store, request = self.make_store(num_blocks=2)
        empty = make_kv(0)
        store.append(request, *empty)
        self.assertEqual(request.block_table, [])
        request.ensure_capacity(8)
        self.assertEqual(request.num_tokens, 0)
        self.assert_kv_equal(store.gather(request), empty)
        store.append(request, *make_kv(5))
        request.release_all()
        self.assertEqual(request.num_tokens, 0)
        self.assert_kv_equal(store.gather(request), empty)
        other = RequestKV(pool, block_size=4)
        kv = make_kv(1, start=300)
        store.append(other, *kv)
        self.assert_kv_equal(store.gather(other), kv)

    def test_invalid_inputs_preserve_state(self):
        pool, store, request = self.make_store()
        k, v = make_kv(2)
        wrong_inputs = [
            (k[0], v[0]),
            (k, v[:1]),
            (k[:, :1], v[:, :1]),
            (k.double(), v.double()),
        ]
        for bad_k, bad_v in wrong_inputs:
            with self.subTest(shape=tuple(bad_k.shape), dtype=bad_k.dtype):
                with self.assertRaises(ValueError):
                    store.append(request, bad_k, bad_v)
        for bad_request in [RequestKV(BlockPool(4), 4), RequestKV(pool, 2)]:
            with self.assertRaises(ValueError):
                store.append(bad_request, k, v)
            with self.assertRaises(ValueError):
                store.gather(bad_request)
            self.assertEqual(bad_request.block_table, [])
            self.assertEqual(bad_request.num_tokens, 0)
        self.assertEqual(request.num_tokens, 0)
        self.assertEqual(request.block_table, [])
        self.assertEqual(list(pool.free_blocks), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
