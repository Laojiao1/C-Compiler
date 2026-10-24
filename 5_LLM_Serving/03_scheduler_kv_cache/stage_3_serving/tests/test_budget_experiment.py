"""对照工具的人工验收：不访问 /proc、GPU 或真实 vLLM 服务。"""

import copy
import unittest
from unittest.mock import patch

from stage_3_serving.budget_experiment import (
    KV_BYTES, MODEL_PATH, REVISION, compare_experiments, expected_options,
    metric_samples, parse_command, run_experiment, summarize_rounds, verify_environment,
)
from stage_3_serving.prepare_mixed_workload import build_requests


def snapshot(label="A"):
    blocks = KV_BYTES // (28672 * 16)
    return {"options": expected_options(label),
            "process": {"pid": 123, "start_ticks": "1000"},
            "cache": {"block_size": "16", "cache_dtype": "auto",
                      "enable_prefix_caching": "False", "kv_cache_memory_bytes": str(KV_BYTES),
                      "num_gpu_blocks": str(blocks), "kv_cache_size_tokens": str(blocks * 16)},
            "packages": {"vllm": "test", "torch": "test", "triton": "test"}}


def round_report(throughput=300, valid=True):
    return {"summary": {"pilot_pass": valid, "output_tokens_per_s": throughput,
                        "groups": {"short": {"mean_client_tpot_ms": 13},
                                   "long": {"mean_client_ttft_ms": 200}}}}


def experiment(label):
    rounds = [round_report(throughput=x) for x in (300, 305, 295)]
    return {"config": label, "measured_rounds": rounds,
            "summary": summarize_rounds(rounds), "error": None,
            "workload_sha256": "same-input", "snapshots": [snapshot(label)]}


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


def workload():
    return {"api_path": "/v1/completions", "requirements": {"enable_prefix_caching": False},
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct", "revision": REVISION,
            "requests": build_requests(FakeTokenizer())}


class BudgetExperimentTests(unittest.TestCase):
    def test_launcher_options_roundtrip(self):
        for label in ("A", "B", "C"):
            expected = expected_options(label)
            argv = ["python", "/env/bin/vllm", "serve", MODEL_PATH]
            for key, value in expected.items():
                if key.startswith("--"):
                    argv.extend([key, value])
            argv += ["--no-enable-prefix-caching",
                     "--enable-chunked-prefill" if expected["chunked_prefill"]
                     else "--no-enable-chunked-prefill"]
            self.assertEqual(parse_command(argv), expected)

    def test_duplicate_unknown_and_missing_options_are_rejected(self):
        base = ["vllm", "serve", MODEL_PATH]
        for tail in (["--port"], ["--port", "--dtype"], ["--config", "hidden.yaml"],
                     ["--enable-chunked-prefill", "--no-enable-chunked-prefill"]):
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                parse_command(base + tail)

    def test_metric_labels_escapes_and_other_series(self):
        raw = ('# HELP unrelated ignored\n'
               'vllm:cache_config_info{engine="0",test="a\\\"b\\nc"} 1.0\n'
               'vllm:cache_config_info_created 42\n')
        self.assertEqual(metric_samples(raw, "vllm:cache_config_info"),
                         [({"engine": "0", "test": 'a"b\nc'}, 1.0)])
        self.assertEqual(metric_samples("some_metric 0\n", "some_metric"), [({}, 0.0)])
        with self.assertRaises(ValueError):
            metric_samples("some_metric NaN", "some_metric")

    def test_fixed_capacity_and_correct_configuration(self):
        for label in ("A", "B", "C"):
            verify_environment(snapshot(label), label)
        self.assertEqual(snapshot()["cache"]["num_gpu_blocks"], "2340")
        self.assertEqual(snapshot()["cache"]["kv_cache_size_tokens"], "37440")

    def test_budget_cache_and_engine_drift_are_rejected(self):
        original = snapshot()
        for key in ("budget", "prefix", "blocks", "restart", "start_time", "cache"):
            changed = copy.deepcopy(original)
            if key == "budget":
                changed["options"]["--max-num-batched-tokens"] = "2048"
            elif key == "prefix":
                changed["cache"]["enable_prefix_caching"] = "True"
            elif key == "blocks":
                changed["cache"]["num_gpu_blocks"] = "2000"
            elif key == "restart":
                changed["process"]["pid"] = 456
            elif key == "start_time":
                changed["process"]["start_ticks"] = "2000"
            else:
                changed["cache"]["extra"] = "changed"
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_environment(changed, "A", original)

    def test_summary_preserves_rounds_and_never_discards_failure(self):
        rounds = [round_report(x) for x in (300, 280, 310)]
        summary = summarize_rounds(rounds)
        self.assertEqual(summary["output_tokens_per_s"],
                         {"round_values": [300, 280, 310], "median": 300, "min": 280, "max": 310})
        rounds[-1]["summary"]["pilot_pass"] = False
        self.assertFalse(summarize_rounds(rounds)["valid"])
        self.assertFalse(summarize_rounds(rounds[:2])["valid"])

    def test_warmup_is_separate_and_each_round_has_configuration_check(self):
        with patch("stage_3_serving.budget_experiment.capture_environment", return_value=snapshot()) as capture, \
             patch("stage_3_serving.budget_experiment.run_mixed", side_effect=[round_report(x) for x in (1, 2, 300, 310, 290)]):
            result = run_experiment("A", workload())
        self.assertEqual(capture.call_count, 6)
        self.assertIsNone(result["error"])
        self.assertEqual(len(result["warmup_rounds"]), 2)
        self.assertEqual(len(result["measured_rounds"]), 3)
        self.assertEqual(result["summary"]["output_tokens_per_s"]["round_values"], [300, 310, 290])

    def test_wrong_service_fails_before_inference(self):
        with patch("stage_3_serving.budget_experiment.capture_environment", return_value=snapshot("B")), \
             patch("stage_3_serving.budget_experiment.run_mixed") as inference:
            result = run_experiment("A", workload())
        inference.assert_not_called()
        self.assertFalse(result["summary"]["valid"])
        self.assertEqual(result["error"]["type"], "ValueError")

    def test_failed_round_is_retained_and_stops_further_measurement(self):
        with patch("stage_3_serving.budget_experiment.capture_environment", return_value=snapshot()), \
             patch("stage_3_serving.budget_experiment.run_mixed", return_value=round_report(valid=False)) as inference:
            result = run_experiment("A", workload())
        self.assertEqual(inference.call_count, 1)
        self.assertEqual(len(result["warmup_rounds"]), 1)
        self.assertFalse(result["summary"]["valid"])

    def test_compare_only_ab_bc_and_check_input_and_environment(self):
        results = [experiment(label) for label in ("A", "B", "C")]
        rows = compare_experiments(results)
        self.assertEqual({r["comparison"] for r in rows}, {"A/B", "B/C"})
        self.assertEqual(len(rows), 6)
        for key in ("input", "version", "failure", "duplicate"):
            changed = copy.deepcopy(results)
            if key == "input":
                changed[2]["workload_sha256"] = "different"
            elif key == "version":
                changed[2]["snapshots"][0]["packages"]["vllm"] = "different"
            elif key == "failure":
                changed[2]["summary"]["valid"] = False
            else:
                changed[2] = experiment("A")
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare_experiments(changed)


if __name__ == "__main__":
    unittest.main()
