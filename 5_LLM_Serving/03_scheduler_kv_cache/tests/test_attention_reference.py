"""对照 PyTorch SDPA，检验 GQA 对应关系及分页存储后的 Attention 输出。

SDPA 口径：一个 Query 可见所有有效历史，is_causal=False、dropout_p=0。
API 依据：https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
"""

import unittest

import torch
import torch.nn.functional as F

from attention_reference import single_query_attention
from paged_kv import BlockPool, RequestKV
from paged_kv_store import PagedKVStore


def sdpa_reference(q, k, v):
    # 独立基准显式展开 KV heads，适配 SDPA 的 [batch, heads, tokens, dim] 布局。
    group_size = q.shape[0] // k.shape[1]
    k_heads = k.repeat_interleave(group_size, dim=1).transpose(0, 1)
    v_heads = v.repeat_interleave(group_size, dim=1).transpose(0, 1)
    output = F.scaled_dot_product_attention(
        q.unsqueeze(0).unsqueeze(2),
        k_heads.unsqueeze(0),
        v_heads.unsqueeze(0),
        dropout_p=0.0,
        is_causal=False,
    )
    return output[0, :, 0, :]


class AttentionReferenceTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(20261020)

    def random_tensor(self, *shape):
        return torch.randn(shape, generator=self.generator, dtype=torch.float32)

    def test_mha_gqa_mqa_against_sdpa(self):
        cases = [(1, 2, 2, 8), (7, 4, 2, 8), (8, 4, 1, 8), (9, 12, 2, 128)]
        for num_tokens, hq, hkv, dim in cases:
            with self.subTest(tokens=num_tokens, hq=hq, hkv=hkv, dim=dim):
                q = self.random_tensor(hq, dim)
                k = self.random_tensor(num_tokens, hkv, dim)
                v = self.random_tensor(num_tokens, hkv, dim)
                result = single_query_attention(q, k, v)
                self.assertEqual(tuple(result.shape), (hq, dim))
                torch.testing.assert_close(
                    result, sdpa_reference(q, k, v), rtol=1e-5, atol=1e-6
                )

    def test_paged_history_matches_original_contiguous_history(self):
        pool = BlockPool(5)
        store = PagedKVStore(pool, block_size=4, num_kv_heads=2, head_dim=8)
        store.k_pool.fill_(float("nan"))
        store.v_pool.fill_(float("nan"))
        request_a = RequestKV(pool, 4)
        request_b = RequestKV(pool, 4)
        q = self.random_tensor(4, 8)
        k_a, v_a = self.random_tensor(7, 2, 8), self.random_tensor(7, 2, 8)
        k_b, v_b = self.random_tensor(4, 2, 8), self.random_tensor(4, 2, 8)
        store.append(request_a, k_a[:4], v_a[:4])
        store.append(request_b, k_b, v_b)
        store.append(request_a, k_a[4:], v_a[4:])
        self.assertEqual(request_a.block_table, [0, 2])
        request_a.ensure_capacity(12)
        self.assertEqual(request_a.num_tokens, 7)
        for request, original_k, original_v in [
            (request_a, k_a, v_a),
            (request_b, k_b, v_b),
        ]:
            with self.subTest(block_table=request.block_table):
                paged_k, paged_v = store.gather(request)
                result = single_query_attention(q, paged_k, paged_v)
                torch.testing.assert_close(
                    result, sdpa_reference(q, original_k, original_v),
                    rtol=1e-5, atol=1e-6,
                )

    def test_zero_query_returns_mean_of_corresponding_value_head(self):
        q = torch.zeros(4, 3)
        k = self.random_tensor(5, 2, 3)
        v = torch.arange(30, dtype=torch.float32).reshape(5, 2, 3)
        expected = v.mean(dim=0).repeat_interleave(2, dim=0)
        torch.testing.assert_close(
            single_query_attention(q, k, v), expected, rtol=1e-5, atol=1e-6
        )

    def test_invalid_inputs(self):
        q = self.random_tensor(4, 8)
        k, v = self.random_tensor(7, 2, 8), self.random_tensor(7, 2, 8)
        cases = [
            (q.unsqueeze(0), k, v),
            (q, k, v[:6]),
            (q[:, :4], k, v),
            (q[:3], k, v),
            (q, k[:0], v[:0]),
            (q.double(), k.double(), v.double()),
        ]
        for inputs in cases:
            with self.subTest(shapes=[tuple(t.shape) for t in inputs]):
                with self.assertRaises(ValueError):
                    single_query_attention(*inputs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
