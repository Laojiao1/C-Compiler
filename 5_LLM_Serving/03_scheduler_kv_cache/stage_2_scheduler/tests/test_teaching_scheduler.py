"""验证到达、接纳、块容量与真实写入之间的区别；不是性能 benchmark。"""

import unittest

import torch

from stage_1_paged_kv.paged_kv import BlockPool, RequestKV
from stage_1_paged_kv.paged_kv_store import PagedKVStore
from stage_2_scheduler.scheduler import SchedulingRequest
from stage_2_scheduler.teaching_scheduler import TeachingScheduler


class TeachingSchedulerTests(unittest.TestCase):
    def make_scheduler(self, blocks=16, budget=8, seqs=3, model_len=32):
        pool = BlockPool(blocks)
        store = PagedKVStore(pool, 4, 1, 2)
        return TeachingScheduler(store, budget, seqs, model_len)

    def make_request(self, scheduler, name="A", prompt=6, limit=2,
                     computed=0, generated=0):
        kv = RequestKV(scheduler.store.pool, 4)
        values = torch.zeros(computed, 1, 2)
        scheduler.store.append(kv, values, values)
        return SchedulingRequest(name, prompt, kv, generated, limit)

    def test_arrival_then_partial_prefill_stays_running(self):
        scheduler = self.make_scheduler(budget=4)
        a = self.make_request(scheduler)
        scheduler.submit(a, arrival_step=1)
        first = scheduler.step()
        self.assertEqual(first["plan"], ())
        self.assertEqual(first["states"], {"A": "not_arrived"})
        second = scheduler.step()
        self.assertEqual(second["used_tokens"], 4)
        self.assertEqual(second["states"], {"A": "running"})
        self.assertEqual((a.kv.num_tokens, a.num_generated_tokens), (4, 0))
        self.assertEqual(second["used_blocks"], 1)

    def test_reservation_does_not_publish_unwritten_kv(self):
        scheduler = self.make_scheduler(blocks=1, budget=4)
        a = self.make_request(scheduler)
        scheduler.submit(a)
        scheduler._collect_arrivals()
        plan = scheduler._reserve_plan()
        self.assertEqual([(p.request_id, p.num_tokens) for p in plan], [("A", 4)])
        self.assertEqual(a.kv.num_tokens, 0)
        self.assertEqual(a.num_generated_tokens, 0)
        self.assertEqual(len(a.kv.block_table), 1)
        self.assertEqual(scheduler.waiting, [])
        self.assertEqual(scheduler.running, [a])

    def test_finished_request_replaced_at_next_iteration(self):
        scheduler = self.make_scheduler(seqs=1)
        a = self.make_request(scheduler, "A", prompt=3, limit=1)
        b = self.make_request(scheduler, "B", prompt=3, limit=1)
        scheduler.submit(a)
        scheduler.submit(b)
        first = scheduler.step()
        self.assertEqual(first["states"], {"A": "finished", "B": "waiting"})
        self.assertEqual(first["used_blocks"], 0)
        self.assertEqual(first["emitted"], ("A",))
        second = scheduler.step()
        self.assertEqual(second["emitted"], ("B",))
        self.assertEqual(second["states"], {"A": "finished", "B": "finished"})

    def test_running_does_not_mean_selected_this_step(self):
        scheduler = self.make_scheduler(budget=2, seqs=2)
        a = self.make_request(scheduler, "A", prompt=1, limit=4)
        b = self.make_request(scheduler, "B", prompt=1, limit=4)
        scheduler.submit(a)
        scheduler.submit(b)
        scheduler.step()
        scheduler.max_num_seqs = 1
        trace = scheduler.step()
        self.assertEqual([p.request_id for p in trace["plan"]], ["A"])
        self.assertEqual(trace["states"], {"A": "running", "B": "running"})
        self.assertEqual(b.num_generated_tokens, 1)

    def test_failed_reservation_preserves_budget_for_later_tail_slot(self):
        scheduler = self.make_scheduler(blocks=3, budget=2, seqs=1)
        a = self.make_request(scheduler, "A", prompt=4, limit=3, computed=4, generated=1)
        b = self.make_request(scheduler, "B", prompt=3, limit=2, computed=3, generated=1)
        # 第三块被额外持有者占用，使 A 无法领取新块；B 的尾块还剩一个位置。
        holder = RequestKV(scheduler.store.pool, 4)
        holder.ensure_capacity(4)
        scheduler.submit(a)
        scheduler.submit(b)
        trace = scheduler.step()
        self.assertEqual(trace["blocked"], ("A",))
        self.assertEqual([(p.request_id, p.num_tokens) for p in trace["plan"]], [("B", 1)])
        self.assertEqual(trace["used_tokens"], 1)
        self.assertEqual((a.kv.num_tokens, a.num_generated_tokens), (4, 1))
        self.assertEqual(trace["states"], {"A": "waiting", "B": "finished"})
        # B 完成本轮才释放的空间，在下一轮可供 A 使用。
        next_trace = scheduler.step()
        self.assertEqual([p.request_id for p in next_trace["plan"]], ["A"])

    def test_two_requests_cannot_reserve_the_same_free_block(self):
        scheduler = self.make_scheduler(blocks=1, budget=6, seqs=2)
        a = self.make_request(scheduler, "A", prompt=3, limit=2)
        b = self.make_request(scheduler, "B", prompt=3, limit=2)
        scheduler.submit(a)
        scheduler.submit(b)
        trace = scheduler.step()
        self.assertEqual([p.request_id for p in trace["plan"]], ["A"])
        self.assertEqual(trace["blocked"], ("B",))
        self.assertEqual(b.kv.block_table, [])
        self.assertEqual(b.kv.num_tokens, 0)
        self.assertEqual(trace["states"], {"A": "running", "B": "waiting"})

    def test_no_preemption_can_stall_even_with_token_budget(self):
        scheduler = self.make_scheduler(blocks=4, budget=14, seqs=2)
        a = self.make_request(scheduler, "A", prompt=7, limit=4)
        b = self.make_request(scheduler, "B", prompt=7, limit=4)
        scheduler.submit(a)
        scheduler.submit(b)
        scheduler.step()  # 各写 7 个输入位置，耗尽 4 块，但各有一个尾部空槽。
        scheduler.step()  # 各写第 8 个位置，仍能在已有块中前进。
        trace = scheduler.step()  # 第 9 个位置各需新块，均无法安排。
        self.assertEqual(trace["plan"], ())
        self.assertEqual(trace["blocked"], ("A", "B"))
        self.assertEqual(trace["used_tokens"], 0)
        self.assertEqual(trace["used_blocks"], 4)
        self.assertEqual(trace["states"], {"A": "running", "B": "running"})
        self.assertEqual((a.kv.num_tokens, b.kv.num_tokens), (8, 8))

    def test_model_len_checks_total_length_without_reserving_it(self):
        scheduler = self.make_scheduler(blocks=1, model_len=8)
        a = self.make_request(scheduler, prompt=6, limit=2)
        scheduler.submit(a)  # 总长度刚好 8，虽然池容量只有 4，登记仍然合法。
        self.assertEqual(a.kv.block_table, [])
        self.assertEqual(len(scheduler.store.pool.free_blocks), 1)
        too_long = self.make_request(scheduler, "B", prompt=6, limit=3)
        with self.assertRaises(ValueError):
            scheduler.submit(too_long)
        self.assertEqual(list(scheduler.requests), ["A"])

    def test_shared_prefix_counts_physical_blocks_once(self):
        scheduler = self.make_scheduler(blocks=3, budget=4)
        source = self.make_request(scheduler, "source", prompt=8, computed=8)
        a = SchedulingRequest("A", 10, source.kv.share_prefix(8), 0, 1)
        b = SchedulingRequest("B", 10, source.kv.share_prefix(8), 0, 1)
        scheduler.submit(a)
        scheduler.submit(b)
        trace = scheduler.step()
        self.assertEqual([p.request_id for p in trace["plan"]], ["A"])
        self.assertEqual(trace["blocked"], ("B",))
        # A 结束后释放自己的引用，source 和 B 仍保持两个共享物理块有效。
        self.assertEqual(trace["used_blocks"], 2)
        self.assertEqual(b.kv.num_tokens, 8)
        second = scheduler.step()
        self.assertEqual(second["emitted"], ("B",))
        source.kv.release_all()
        self.assertEqual(len(scheduler.store.pool.free_blocks), 3)

    def test_trace_snapshots_do_not_change_with_later_steps(self):
        scheduler = self.make_scheduler(budget=4)
        a = self.make_request(scheduler)
        scheduler.submit(a)
        first = scheduler.step()
        scheduler.step()
        scheduler.step()
        self.assertTrue(a.finished)
        self.assertEqual(first["states"], {"A": "running"})
        self.assertEqual(first["kv_tokens"], {"A": 4})
        self.assertEqual(first["block_tables"], {"A": (0,)})

    def test_invalid_registration_does_not_change_queues(self):
        scheduler = self.make_scheduler()
        a = self.make_request(scheduler)
        scheduler.submit(a)
        with self.assertRaises(ValueError):
            scheduler.submit(a)
        scheduler.step()
        b = self.make_request(scheduler, "B")
        with self.assertRaises(ValueError):
            scheduler.submit(b, arrival_step=0)
        self.assertEqual(list(scheduler.requests), ["A"])
        self.assertEqual(scheduler.waiting, [])
        self.assertEqual(scheduler.running, [a])


if __name__ == "__main__":
    unittest.main()
