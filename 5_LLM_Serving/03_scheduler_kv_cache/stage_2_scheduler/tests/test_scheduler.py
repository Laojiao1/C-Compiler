"""只验证单步选择及只读性，不测模型性能或模拟完整请求生命周期。"""

import unittest

import torch

from stage_1_paged_kv.paged_kv import BlockPool, RequestKV
from stage_1_paged_kv.paged_kv_store import PagedKVStore
from stage_2_scheduler.scheduler import ScheduledChunk, SchedulingRequest, plan_step


class SchedulerPlanTests(unittest.TestCase):
    def make_requests(self, c_prompt=10, c_computed=0):
        pool = BlockPool(32)
        store = PagedKVStore(pool, 4, 1, 2)

        def make(request_id, prompt_len, computed, generated):
            kv = RequestKV(pool, 4)
            # 用原有 append 真正写入 CPU K/V，避免把容量预留冒充有效历史。
            values = torch.zeros(computed, 1, 2, dtype=torch.float32)
            store.append(kv, values, values)
            return SchedulingRequest(request_id, prompt_len, kv, generated)

        a = make("A", 8, 8, 1)
        b = make("B", 8, 8, 1)
        c = make("C", c_prompt, c_computed, 0)
        # 故意把 Prefill 放在前面，检查优先级不是偶然由输入顺序满足。
        return pool, store, [c, a, b]

    def test_decode_priority_and_chunked_prefill(self):
        _, _, requests = self.make_requests()
        self.assertEqual(plan_step(requests, 8, 3), [
            ScheduledChunk("A", "decode", 8, 1),
            ScheduledChunk("B", "decode", 8, 1),
            ScheduledChunk("C", "prefill", 0, 6),
        ])

    def test_smaller_budget_leaves_two_prefill_positions(self):
        _, _, requests = self.make_requests()
        self.assertEqual(plan_step(requests, 4, 3), [
            ScheduledChunk("A", "decode", 8, 1),
            ScheduledChunk("B", "decode", 8, 1),
            ScheduledChunk("C", "prefill", 0, 2),
        ])

    def test_sequence_limit_excludes_third_request(self):
        _, _, requests = self.make_requests()
        self.assertEqual(plan_step(requests, 8, 2), [
            ScheduledChunk("A", "decode", 8, 1),
            ScheduledChunk("B", "decode", 8, 1),
        ])

    def test_one_token_budget_keeps_decode_input_order(self):
        _, _, requests = self.make_requests()
        self.assertEqual(plan_step(requests, 1, 3), [
            ScheduledChunk("A", "decode", 8, 1),
        ])

    def test_partial_prefill_uses_original_positions_and_can_leave_budget(self):
        _, _, requests = self.make_requests(c_computed=6)
        # C 还剩 4 个输入位置；不能为了填满预算而提前计算未知输出。
        self.assertEqual(plan_step(requests, 8, 3), [
            ScheduledChunk("A", "decode", 8, 1),
            ScheduledChunk("B", "decode", 8, 1),
            ScheduledChunk("C", "prefill", 6, 4),
        ])

    def test_empty_or_no_pending_work(self):
        self.assertEqual(plan_step([], 8, 3), [])
        _, _, requests = self.make_requests()
        a = requests[1]
        # 仅用于验证选择器的过滤行为：已有输入 KV，但没有待算输出。
        # 正常执行器应在完成 Prefill 后采样，或将完成请求移出候选集合。
        a.num_generated_tokens = 0
        self.assertEqual(plan_step([a], 8, 3), [])

    def test_planning_does_not_modify_requests_pool_or_storage(self):
        pool, store, requests = self.make_requests(c_computed=6)

        def snapshot():
            return (
                [id(r) for r in requests],
                [(r.request_id, r.prompt_len, r.num_generated_tokens,
                  r.kv.num_tokens, list(r.kv.block_table)) for r in requests],
                list(pool.free_blocks), list(pool.ref_counts),
            )

        # 初始化整池后取快照，避免未写槽位的 NaN 干扰精确比较。
        store.k_pool.fill_(17)
        store.v_pool.fill_(29)
        before = snapshot()
        k_before, v_before = store.k_pool.clone(), store.v_pool.clone()
        plan = plan_step(requests, 8, 3)
        self.assertEqual(plan, plan_step(requests, 8, 3))
        self.assertEqual(snapshot(), before)
        torch.testing.assert_close(store.k_pool, k_before, rtol=0, atol=0)
        torch.testing.assert_close(store.v_pool, v_before, rtol=0, atol=0)

    def test_nonpositive_limits_are_rejected(self):
        for budget, seqs in [(0, 3), (-1, 3), (8, 0), (8, -1)]:
            with self.subTest(budget=budget, seqs=seqs):
                with self.assertRaises(ValueError):
                    plan_step([], budget, seqs)


if __name__ == "__main__":
    unittest.main()
