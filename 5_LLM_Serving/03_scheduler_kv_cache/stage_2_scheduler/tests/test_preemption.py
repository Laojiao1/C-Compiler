"""显式抢占与分块重算：不重复产生输出，不丢恢复终点，不误回收共享块。"""

import unittest

import torch

from stage_1_paged_kv.paged_kv import BlockPool, RequestKV
from stage_1_paged_kv.paged_kv_store import PagedKVStore
from stage_2_scheduler.scheduler import ScheduledChunk, SchedulingRequest, plan_step
from stage_2_scheduler.scheduler_simulator import execute_chunk
from stage_2_scheduler.teaching_scheduler import TeachingScheduler


class PreemptionTests(unittest.TestCase):
    def make_scheduler(self, blocks=8, budget=4):
        store = PagedKVStore(BlockPool(blocks), 4, 1, 2)
        return TeachingScheduler(store, budget, 2, 32)

    def make_request(self, scheduler, name="A", prompt=7, limit=4):
        return SchedulingRequest(name, prompt, RequestKV(scheduler.store.pool, 4),
                                 max_new_tokens=limit)

    def make_recovering(self, scheduler):
        # 输入 8，已输出 2；抢占前的有效 KV 包含输入和第一个输出，共 9 个位置。
        return SchedulingRequest("B", 8, RequestKV(scheduler.store.pool, 4),
                                 num_generated_tokens=2, max_new_tokens=4,
                                 recompute_until=9)

    def test_preempt_preserves_output_count_and_moves_to_waiting(self):
        scheduler = self.make_scheduler()
        request = self.make_request(scheduler)
        scheduler.submit(request)
        scheduler.step()  # 部分 Prefill，有 4 个有效位置。
        event = scheduler.preempt("A")
        self.assertEqual(event, {"request_id": "A", "dropped_kv_tokens": 4,
                                 "freed_blocks": 1})
        self.assertEqual(request.recompute_until, 4)
        self.assertEqual(request.kv.num_tokens, 0)
        self.assertEqual(request.kv.block_table, [])
        self.assertEqual(request.num_generated_tokens, 0)
        self.assertFalse(request.finished)
        self.assertEqual(scheduler.running, [])
        self.assertEqual(scheduler.waiting, [request])

    def test_chunked_recompute_emits_nothing_then_resumes_decode(self):
        scheduler = self.make_scheduler()
        request = self.make_recovering(scheduler)
        for start, count in [(0, 4), (4, 4), (8, 1)]:
            plan = plan_step([request], 4, 1)
            self.assertEqual(plan, [ScheduledChunk("B", "recompute", start, count)])
            self.assertFalse(execute_chunk(request, plan[0], scheduler.store))
            self.assertEqual(request.num_generated_tokens, 2)
            self.assertFalse(request.finished)
        self.assertEqual(request.recompute_until, 0)
        self.assertEqual(request.num_recomputed_tokens, 9)
        self.assertEqual(request.kv.num_tokens, 9)
        plan = plan_step([request], 4, 1)
        self.assertEqual(plan, [ScheduledChunk("B", "decode", 9, 1)])
        self.assertTrue(execute_chunk(request, plan[0], scheduler.store))
        self.assertEqual((request.kv.num_tokens, request.num_generated_tokens), (10, 3))

    def test_recompute_stops_at_old_endpoint_even_with_large_budget(self):
        scheduler = self.make_scheduler(budget=20)
        request = self.make_recovering(scheduler)
        self.assertEqual(plan_step([request], 20, 1),
                         [ScheduledChunk("B", "recompute", 0, 9)])
        with self.assertRaises(ValueError):
            execute_chunk(request, ScheduledChunk("B", "recompute", 0, 10), scheduler.store)
        self.assertEqual(request.kv.num_tokens, 0)

    def test_repeated_preemption_keeps_original_target_and_counts_wasted_work(self):
        scheduler = self.make_scheduler(budget=4)
        request = self.make_request(scheduler, prompt=8)
        scheduler.submit(request)
        scheduler.step()
        scheduler.step()  # 输入完成，KV=8，已有首个输出。
        scheduler.preempt("A")
        scheduler.step()  # 重算前 4 个位置，原恢复终点仍是 8。
        self.assertEqual(request.num_recomputed_tokens, 4)
        scheduler.preempt("A")
        self.assertEqual(request.recompute_until, 8)
        scheduler.step()
        scheduler.step()
        self.assertEqual(request.kv.num_tokens, 8)
        self.assertEqual(request.recompute_until, 0)
        self.assertEqual(request.num_generated_tokens, 1)
        self.assertEqual(request.num_recomputed_tokens, 12)

    def test_partial_prefill_recovery_does_not_emit_first_output_early(self):
        scheduler = self.make_scheduler(budget=4)
        request = self.make_request(scheduler, prompt=7)
        scheduler.submit(request)
        scheduler.step()
        scheduler.preempt("A")
        recovery = scheduler.step()
        self.assertEqual(recovery["recomputed_tokens"], 4)
        self.assertEqual(recovery["emitted"], ())
        self.assertEqual((request.kv.num_tokens, request.num_generated_tokens), (4, 0))
        finish_prefill = scheduler.step()
        self.assertEqual(finish_prefill["plan"], (ScheduledChunk("A", "prefill", 4, 3),))
        self.assertEqual(finish_prefill["emitted"], ("A",))

    def test_shared_holder_can_prevent_any_physical_block_from_being_freed(self):
        scheduler = self.make_scheduler()
        request = self.make_request(scheduler, prompt=4)
        scheduler.submit(request)
        scheduler.step()
        survivor = request.kv.share_prefix(4)
        k_before, v_before = scheduler.store.gather(survivor)
        event = scheduler.preempt("A")
        self.assertEqual(event["freed_blocks"], 0)
        k, v = scheduler.store.gather(survivor)
        torch.testing.assert_close(k, k_before, rtol=0, atol=0)
        torch.testing.assert_close(v, v_before, rtol=0, atol=0)
        survivor.release_all()
        self.assertEqual(len(scheduler.store.pool.free_blocks), scheduler.store.pool.num_blocks)

    def test_explicit_preemption_breaks_the_four_block_stall(self):
        scheduler = self.make_scheduler(blocks=4, budget=14)
        a = self.make_request(scheduler, "A")
        b = self.make_request(scheduler, "B")
        scheduler.submit(a)
        scheduler.submit(b)
        scheduler.step()
        scheduler.step()
        stalled = scheduler.step()
        self.assertEqual(stalled["plan"], ())
        self.assertEqual((a.kv.num_tokens, b.kv.num_tokens), (8, 8))
        event = scheduler.preempt("B")
        self.assertEqual(event["freed_blocks"], 2)
        self.assertEqual(b.num_generated_tokens, 2)
        resumed = scheduler.step()
        self.assertEqual(resumed["plan"], (ScheduledChunk("A", "decode", 8, 1),))
        self.assertEqual(resumed["preemptions"], (event,))
        scheduler.step()  # A 完成，归还它的全部块。
        self.assertTrue(a.finished)
        rebuild = scheduler.step()
        self.assertEqual(rebuild["plan"], (ScheduledChunk("B", "recompute", 0, 8),))
        self.assertEqual(rebuild["emitted"], ())
        self.assertEqual(b.num_generated_tokens, 2)
        self.assertEqual(rebuild["preemptions"], ())
        scheduler.step()
        scheduler.step()
        self.assertTrue(a.finished and b.finished)
        self.assertEqual((a.num_generated_tokens, b.num_generated_tokens), (4, 4))
        self.assertEqual(b.num_recomputed_tokens, 8)
        self.assertEqual(rebuild["total_recomputed_tokens"], {"A": 0, "B": 8})
        self.assertEqual(len(scheduler.store.pool.free_blocks), 4)

    def test_invalid_preemption_does_not_move_queues(self):
        scheduler = self.make_scheduler()
        request = self.make_request(scheduler)
        scheduler.submit(request, arrival_step=1)
        with self.assertRaises(ValueError):
            scheduler.preempt("A")
        with self.assertRaises(KeyError):
            scheduler.preempt("missing")
        self.assertEqual(scheduler.waiting, [])
        self.assertEqual(scheduler.running, [])
        self.assertEqual(request.recompute_until, 0)

    def test_capacity_failure_during_recompute_does_not_change_counters(self):
        scheduler = self.make_scheduler(blocks=1)
        request = self.make_recovering(scheduler)
        execute_chunk(request, ScheduledChunk("B", "recompute", 0, 4), scheduler.store)
        with self.assertRaises(RuntimeError):
            execute_chunk(request, ScheduledChunk("B", "recompute", 4, 4), scheduler.store)
        self.assertEqual(request.kv.num_tokens, 4)
        self.assertEqual(request.num_recomputed_tokens, 4)
        self.assertEqual(request.recompute_until, 9)
        self.assertEqual(request.num_generated_tokens, 2)


if __name__ == "__main__":
    unittest.main()
