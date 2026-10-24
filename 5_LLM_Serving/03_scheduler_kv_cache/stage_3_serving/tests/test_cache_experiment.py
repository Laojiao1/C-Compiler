"""剩余两组辅助采集的人工验收，不使用真实模型或 GPU。"""

import copy
import unittest
from unittest.mock import patch

from stage_3_serving.budget_experiment import expected_options
from stage_3_serving.cache_experiment import (
    build_workload, check_prefix_delta, counters, run_batch,
    summarize_phases, synchronized_metrics, verify_cache_environment,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def snapshot(mode):
    kv = 134217728 if mode == "capacity" else 1073741824
    options = expected_options("A")
    options.update({"--kv-cache-memory-bytes": str(kv), "prefix_caching": mode == "prefix_on"})
    blocks = kv // (28672 * 16)
    return {"options": options, "cache": {"block_size": "16", "cache_dtype": "auto",
            "kv_cache_memory_bytes": str(kv), "num_gpu_blocks": str(blocks),
            "kv_cache_size_tokens": str(blocks * 16), "enable_prefix_caching": str(mode == "prefix_on")},
            "process": {"pid": 1, "start_ticks": "100"}, "packages": {"vllm": "test"}}


def metrics(success):
    return "\n".join(f"vllm:{name} {success if name == 'request_success_total' else 0}"
                     for name in ("request_success_total", "num_preemptions_total", "prefix_cache_queries_total",
                                  "prefix_cache_hits_total", "generation_tokens_total", "prompt_tokens_total"))


class CacheExperimentTests(unittest.TestCase):
    def test_capacity_boundary_math_and_shared_subset(self):
        workload = build_workload(FakeTokenizer())
        self.assertEqual(snapshot("capacity")["cache"]["num_gpu_blocks"], "292")
        self.assertEqual(snapshot("capacity")["cache"]["kv_cache_size_tokens"], "4672")
        self.assertEqual((1536 + 128 - 1 + 15) // 16, 104)
        self.assertLessEqual(2 * 104, 292)
        self.assertGreater(3 * 104, 292)
        self.assertEqual(len(workload["capacity"]), 3)
        self.assertTrue(all(len(s["body"]["prompt"]) == 1536 for s in workload["capacity"]))

    def test_prefix_targets_are_cold_and_suffixes_do_not_share_full_blocks(self):
        workload = build_workload(FakeTokenizer())
        roots = [group[0]["body"]["prompt"][:16] for group in workload["prefix_warmups"]]
        for block in workload["prefix_rounds"]:
            cold = block["cold"][0]["body"]["prompt"]
            primer = block["primer"][0]["body"]["prompt"]
            single = block["warm_single"][0]["body"]["prompt"]
            roots += [cold[:16], primer[:16]]
            self.assertEqual(len(primer), 1040)
            self.assertEqual(primer[:1024], single[:1024])
            suffixes = [primer[1024:1040], single[1024:1040]]
            for spec in block["warm_batch"]:
                self.assertEqual(spec["body"]["prompt"][:1024], primer[:1024])
                self.assertEqual(len(spec["body"]["prompt"]), 1280)
                suffixes.append(spec["body"]["prompt"][1024:1040])
            self.assertEqual(len({tuple(s) for s in suffixes}), 6)
        self.assertEqual(len({tuple(r) for r in roots}), 8)

    def test_environment_profiles_and_restart_rejection(self):
        for mode in ("capacity", "prefix_off", "prefix_on"):
            original = snapshot(mode)
            verify_cache_environment(original, mode)
            changed = copy.deepcopy(original)
            changed["process"]["pid"] = 2
            with self.assertRaises(ValueError):
                verify_cache_environment(changed, mode, original)
        with self.assertRaises(ValueError):
            verify_cache_environment(snapshot("prefix_on"), "prefix_off")

    def test_wrong_actual_capacity_is_rejected(self):
        changed = snapshot("capacity")
        changed["cache"]["num_gpu_blocks"] = "300"
        with self.assertRaises(ValueError):
            verify_cache_environment(changed, "capacity")

    def test_prefix_counts_are_token_weighted_and_no_query_is_not_zero_percent(self):
        self.assertIsNone(check_prefix_delta("prefix_off", "warm_batch", {
            "prefix_cache_queries_total": 0, "prefix_cache_hits_total": 0}))
        self.assertEqual(check_prefix_delta("prefix_on", "warm_batch", {
            "prefix_cache_queries_total": 5120, "prefix_cache_hits_total": 4096}), 0.8)
        with self.assertRaises(ValueError):
            check_prefix_delta("prefix_on", "cold", {"prefix_cache_queries_total": 1280, "prefix_cache_hits_total": 16})
        with self.assertRaises(ValueError):
            check_prefix_delta("prefix_off", "warm_batch", {"prefix_cache_queries_total": 5120, "prefix_cache_hits_total": 4096})

    def test_missing_counter_is_not_treated_as_zero(self):
        self.assertEqual(counters(metrics(2))["request_success_total"], 2)
        with self.assertRaises(ValueError):
            counters("vllm:request_success_total 2")

    def test_counter_sync_waits_for_completed_requests(self):
        with patch("stage_3_serving.cache_experiment.read_metrics", side_effect=[metrics(1), metrics(2)]), \
             patch("stage_3_serving.cache_experiment.time.sleep") as sleep:
            self.assertEqual(synchronized_metrics(2), metrics(2))
        sleep.assert_called_once()
        with patch("stage_3_serving.cache_experiment.read_metrics", return_value=metrics(3)):
            with self.assertRaises(ValueError):
                synchronized_metrics(2)

    def test_batch_uses_actual_output_count_and_full_http_window(self):
        def fake_measure(url, body, *args, **kwargs):
            return {"measurement_valid": True, "finish_reason": "length", "t0_perf_counter_s": 100,
                    "times_s": {"done": 1}, "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": body["max_tokens"]},
                    "metrics": {"client_ttft_ms": 100, "client_e2e_ms": 1000}}
        specs = build_workload(FakeTokenizer())["capacity"]
        with patch("stage_3_serving.cache_experiment.measure_request", side_effect=fake_measure):
            result = run_batch(specs)
        self.assertTrue(result["summary"]["valid"])
        self.assertEqual(result["summary"]["output_tokens_per_s"], 384)
        self.assertEqual(len(result["requests"]), 3)

    def test_warmups_excluded_and_primer_kept_separately(self):
        def phase(name, value):
            return {"name": name, "batch": {"summary": {"mean_client_ttft_ms": value,
                    "mean_client_e2e_ms": value, "output_tokens_per_s": value}},
                    "counter_delta": {"num_preemptions_total": 0, "prefix_cache_hits_total": 0, "prefix_cache_queries_total": 0}}
        summary = summarize_phases([phase("warmup_prefix", 999), phase("primer", 1),
                                    phase("cold", 200), phase("warm_single", 100)])
        self.assertNotIn("warmup_prefix", summary)
        self.assertEqual(summary["primer"]["mean_client_ttft_ms"]["median"], 1)
        self.assertEqual(summary["cold"]["mean_client_ttft_ms"]["median"], 200)


if __name__ == "__main__":
    unittest.main()
