"""验证计划执行后的状态反馈、连续推进、结束回收和失败边界。"""

import unittest

import torch

from stage_1_paged_kv.paged_kv import BlockPool, RequestKV
from stage_1_paged_kv.paged_kv_store import PagedKVStore
from stage_2_scheduler.scheduler import ScheduledChunk, SchedulingRequest, plan_step
from stage_2_scheduler.scheduler_simulator import execute_chunk


class SchedulerSimulatorTests(unittest.TestCase):
    def make_environment(self, num_blocks=32):
        pool = BlockPool(num_blocks)
        store = PagedKVStore(pool, 4, 1, 2)
        return pool, store

    def make_request(self, store, name="C", prompt=10, computed=0,
                     generated=0, limit=4):
        kv = RequestKV(store.pool, store.block_size)
        values = torch.zeros(computed, store.num_kv_heads, store.head_dim)
        store.append(kv, values, values)
        return SchedulingRequest(name, prompt, kv, generated, limit)

    def run_plan(self, requests, store, budget=8, seqs=3):
        plan = plan_step(requests, budget, seqs)
        by_id = {r.request_id: r for r in requests}
        emitted = []
        for chunk in plan:
            if execute_chunk(by_id[chunk.request_id], chunk, store):
                emitted.append(chunk.request_id)
        return plan, emitted

    def snapshot(self, request, pool):
        return (request.kv.num_tokens, list(request.kv.block_table),
                request.num_generated_tokens, request.finished,
                list(pool.free_blocks), list(pool.ref_counts))

    def test_partial_prefill_appends_without_output(self):
        _, store = self.make_environment()
        request = self.make_request(store)
        result = execute_chunk(request, ScheduledChunk("C", "prefill", 0, 6), store)
        self.assertFalse(result)
        self.assertEqual(request.kv.num_tokens, 6)
        self.assertEqual(request.num_generated_tokens, 0)
        self.assertEqual(request.num_pending_tokens, 4)
        self.assertFalse(request.finished)

    def test_prefill_completion_emits_first_output_without_its_kv(self):
        _, store = self.make_environment()
        request = self.make_request(store, computed=6)
        self.assertTrue(execute_chunk(
            request, ScheduledChunk("C", "prefill", 6, 4), store))
        self.assertEqual(request.kv.num_tokens, 10)
        self.assertEqual(request.num_generated_tokens, 1)
        self.assertEqual(request.num_pending_tokens, 1)
        k, v = store.gather(request.kv)
        self.assertEqual(k.shape, (10, 1, 2))
        # 占位数据按绝对位置构造，检查第二块 Prefill 没有错误地从 0 编号。
        torch.testing.assert_close(
            k[6:], torch.arange(6, 10, dtype=torch.float32).view(4, 1, 1).expand(4, 1, 2))
        torch.testing.assert_close(v[6:], k[6:] + 1000)

    def test_decode_consumes_previous_output_and_emits_next(self):
        _, store = self.make_environment()
        request = self.make_request(store, computed=10, generated=1)
        self.assertTrue(execute_chunk(
            request, ScheduledChunk("C", "decode", 10, 1), store))
        self.assertEqual(request.kv.num_tokens, 11)
        self.assertEqual(request.num_generated_tokens, 2)
        self.assertEqual(request.num_pending_tokens, 1)

    def test_output_limit_finishes_without_computing_last_output_kv(self):
        pool, store = self.make_environment()
        request = self.make_request(store, prompt=3, limit=1)
        self.assertTrue(execute_chunk(
            request, ScheduledChunk("C", "prefill", 0, 3), store))
        self.assertEqual(request.num_generated_tokens, 1)
        self.assertTrue(request.finished)
        self.assertEqual(request.kv.num_tokens, 0)
        self.assertEqual(request.kv.block_table, [])
        self.assertEqual(len(pool.free_blocks), pool.num_blocks)
        # 完成后有效长度归零，但不能被误当成需要重新 Prefill 的活动请求。
        self.assertEqual(plan_step([request], 8, 3), [])
        with self.assertRaises(ValueError):
            execute_chunk(request, ScheduledChunk("C", "prefill", 0, 3), store)

    def test_three_step_trace_and_release_preserve_surviving_request(self):
        pool, store = self.make_environment()
        a = self.make_request(store, "A", 8, 8, 1)
        b = self.make_request(store, "B", 8, 8, 1)
        c = self.make_request(store)
        requests = [c, a, b]
        trace = []
        for _ in range(3):
            plan, emitted = self.run_plan(requests, store)
            trace.append(([(p.request_id, p.num_tokens) for p in plan], emitted))
        self.assertEqual(trace, [
            ([("A", 1), ("B", 1), ("C", 6)], ["A", "B"]),
            ([("A", 1), ("B", 1), ("C", 4)], ["A", "B", "C"]),
            ([("C", 1), ("A", 1), ("B", 1)], ["C", "A", "B"]),
        ])
        self.assertTrue(a.finished and b.finished)
        self.assertFalse(c.finished)
        self.assertEqual((c.kv.num_tokens, c.num_generated_tokens), (11, 2))
        self.assertEqual(len(pool.free_blocks), pool.num_blocks - 3)
        k, _ = store.gather(c.kv)
        torch.testing.assert_close(
            k, torch.arange(11, dtype=torch.float32).view(11, 1, 1).expand(11, 1, 2))

    def test_new_request_can_join_next_iteration_after_completion(self):
        pool, store = self.make_environment()
        a = self.make_request(store, "A", 3, 3, 1, limit=2)
        c = self.make_request(store, "C", 8, 8, 1, limit=4)
        requests = [a, c]
        self.run_plan(requests, store, seqs=2)
        self.assertTrue(a.finished)
        # 只在迭代边界加入 D，不在已经形成的计划中临时插入。
        d = self.make_request(store, "D", prompt=3, limit=2)
        requests.append(d)
        plan, emitted = self.run_plan(requests, store, seqs=2)
        self.assertEqual([(p.request_id, p.num_tokens) for p in plan],
                         [("C", 1), ("D", 3)])
        self.assertEqual(emitted, ["C", "D"])
        self.run_plan(requests, store, seqs=2)
        self.assertTrue(all(r.finished for r in requests))
        self.assertEqual(len(pool.free_blocks), pool.num_blocks)
        self.assertEqual(self.run_plan(requests, store), ([], []))

    def test_stale_plan_is_rejected_without_state_changes(self):
        pool, store = self.make_environment()
        request = self.make_request(store)
        chunk = ScheduledChunk("C", "prefill", 0, 6)
        execute_chunk(request, chunk, store)
        before = self.snapshot(request, pool)
        with self.assertRaises(ValueError):
            execute_chunk(request, chunk, store)
        self.assertEqual(self.snapshot(request, pool), before)

    def test_invalid_chunks_preserve_state(self):
        pool, store = self.make_environment()
        request = self.make_request(store)
        before = self.snapshot(request, pool)
        for chunk in [ScheduledChunk("other", "prefill", 0, 2),
                      ScheduledChunk("C", "decode", 0, 1),
                      ScheduledChunk("C", "prefill", 0, 0),
                      ScheduledChunk("C", "prefill", 0, 11)]:
            with self.subTest(chunk=chunk), self.assertRaises(ValueError):
                execute_chunk(request, chunk, store)
            self.assertEqual(self.snapshot(request, pool), before)

    def test_capacity_failure_does_not_advance_or_emit(self):
        pool, store = self.make_environment(num_blocks=1)
        request = self.make_request(store)
        execute_chunk(request, ScheduledChunk("C", "prefill", 0, 4), store)
        before = self.snapshot(request, pool)
        k_before, v_before = store.gather(request.kv)
        with self.assertRaises(RuntimeError):
            execute_chunk(request, ScheduledChunk("C", "prefill", 4, 2), store)
        self.assertEqual(self.snapshot(request, pool), before)
        k, v = store.gather(request.kv)
        torch.testing.assert_close(k, k_before, rtol=0, atol=0)
        torch.testing.assert_close(v, v_before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
