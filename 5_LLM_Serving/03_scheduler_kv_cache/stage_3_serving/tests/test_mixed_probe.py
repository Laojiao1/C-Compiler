"""固定混合试验的辅助工具验收；人工时间和假服务不是 GPU 性能数据。"""

import copy
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from stage_3_serving.mixed_probe import run_mixed, summarize, validate_workload
from stage_3_serving.prepare_mixed_workload import build_requests
from stage_3_serving.stream_probe import StreamCapture, measure_request


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


def workload():
    return {"api_path": "/v1/completions", "requirements": {"enable_prefix_caching": False},
            "requests": build_requests(FakeTokenizer())}


def text_chunk(text, finish=None):
    return json.dumps({"choices": [{"index": 0, "text": text,
                                    "finish_reason": finish}]}, ensure_ascii=False)


def usage_chunk(prompt, output):
    return json.dumps({"choices": [], "usage": {"prompt_tokens": prompt,
                      "completion_tokens": output, "total_tokens": prompt + output}})


def synthetic_reports():
    reports = []
    for index, spec in enumerate(workload()["requests"]):
        short = spec["cohort"] == "short"
        start = 100 + index * 0.001 if short else 100.3 + (index - 4) * 0.01
        first, last, done = (0.1, 0.9, 1.0) if short else (0.2, 0.3, 0.4)
        output = spec["expected_completion_tokens"]
        reports.append({key: spec[key] for key in
                        ("request_id", "cohort", "expected_prompt_tokens", "expected_completion_tokens")} |
                       {"measurement_valid": True, "t0_perf_counter_s": start,
                        "times_s": {"first_content": first, "last_content": last, "done": done},
                        "usage": {"prompt_tokens": spec["expected_prompt_tokens"],
                                  "completion_tokens": output},
                        "metrics": {"client_ttft_ms": first * 1000,
                                    "client_tpot_ms": (done - first) * 1000 / (output - 1)}})
    return reports


class MixedProbeTests(unittest.TestCase):
    def test_completion_text_and_usage_are_parsed(self):
        capture = StreamCapture("completion")
        capture.consume(text_chunk("正文"), 0.1)
        capture.consume(text_chunk("", "length"), 0.2)
        capture.consume(usage_chunk(128, 256), 0.21)
        capture.consume("[DONE]", 0.22)
        report = capture.report()
        self.assertTrue(report["measurement_valid"])
        self.assertEqual(report["text"], "正文")
        self.assertEqual(report["times_s"]["last_content"], 0.1)
        self.assertEqual(report["usage"]["completion_tokens"], 256)

    def test_response_formats_cannot_be_silently_mixed(self):
        with self.assertRaises(ValueError):
            StreamCapture("chat").consume(text_chunk("x"), 0.1)
        with self.assertRaises(ValueError):
            StreamCapture("completion").consume(
                json.dumps({"choices": [{"index": 0, "delta": {"content": "x"}}]}), 0.1)
        with self.assertRaises(ValueError):
            StreamCapture("unknown")

    def test_valid_synthetic_overlap_and_full_batch_window(self):
        summary = summarize(synthetic_reports(), True)
        self.assertTrue(summary["pilot_pass"])
        self.assertEqual(len(summary["client_overlap_pairs"]), 12)
        self.assertEqual(summary["invalid_request_ids"], [])
        self.assertEqual(summary["groups"]["short"]["valid"], 4)
        self.assertEqual(summary["groups"]["long"]["valid"], 3)
        self.assertAlmostEqual(summary["batch_http_window_s"], 1.003)
        self.assertAlmostEqual(summary["output_tokens_per_s"], 1120 / 1.003)

    def test_done_tail_alone_does_not_count_as_generation_overlap(self):
        reports = synthetic_reports()
        for report in reports[:4]:
            report["times_s"]["last_content"] = 0.15
        summary = summarize(reports, True)
        self.assertTrue(summary["all_requests_valid_and_exact_length"])
        self.assertFalse(summary["pilot_pass"])
        self.assertEqual(summary["client_overlap_pairs"], [])

    def test_touching_window_endpoints_is_not_positive_overlap(self):
        reports = synthetic_reports()
        for report in reports[:4]:
            report["times_s"]["last_content"] = 100.5 - report["t0_perf_counter_s"]
        for report in reports[4:]:
            report["t0_perf_counter_s"] = 100.5
        self.assertFalse(summarize(reports, True)["pilot_pass"])

    def test_length_mismatch_keeps_request_and_disables_batch_throughput(self):
        reports = synthetic_reports()
        reports[0]["usage"]["completion_tokens"] = 10
        summary = summarize(reports, True)
        self.assertFalse(summary["pilot_pass"])
        self.assertEqual(summary["invalid_request_ids"], ["short_0"])
        self.assertIsNone(summary["output_tokens_per_s"])
        self.assertEqual(summary["groups"]["short"]["total"], 4)
        self.assertIsNone(summary["groups"]["short"]["mean_client_tpot_ms"])

    def test_sending_long_before_a_short_first_content_is_rejected(self):
        reports = synthetic_reports()
        reports[0]["times_s"]["first_content"] = 0.4
        self.assertFalse(summarize(reports, True)["all_short_first_content_before_long_sends"])

    def test_invalid_workload_is_rejected_before_any_requests(self):
        good = workload()
        validate_workload(good)
        for mutation in ("duplicate", "length", "endpoint", "prefix", "eos"):
            bad = copy.deepcopy(good)
            if mutation == "duplicate":
                bad["requests"][1]["request_id"] = bad["requests"][0]["request_id"]
            elif mutation == "length":
                bad["requests"][0]["body"]["prompt"].pop()
            elif mutation == "endpoint":
                bad["api_path"] = "/v1/chat/completions"
            elif mutation == "prefix":
                bad["requirements"]["enable_prefix_caching"] = True
            else:
                bad["requests"][0]["body"]["ignore_eos"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_workload(bad)

    def test_failed_short_gate_never_sends_long_requests(self):
        def failed_request(*args, **kwargs):
            report = StreamCapture("completion").report({"type": "TestFailure", "message": "test"})
            report["t0_perf_counter_s"] = time.perf_counter()
            return report

        with patch("stage_3_serving.mixed_probe.measure_request", side_effect=failed_request) as mocked:
            result = run_mixed(workload(), gate_timeout_s=1)
        self.assertFalse(result["first_content_gate"]["opened"])
        self.assertEqual(mocked.call_count, 4)
        self.assertEqual(sum(r.get("not_sent", False) for r in result["requests"]), 3)
        self.assertFalse(result["summary"]["pilot_pass"])

    def test_http_callback_runs_once_for_multiple_completion_chunks(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for payload in [text_chunk("a"), text_chunk("b", "length"),
                                usage_chunk(2, 2), "[DONE]"]:
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()

            def log_message(self, *args):
                pass

        calls = []
        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                report = measure_request(f"http://127.0.0.1:{server.server_port}/v1/completions",
                                         {}, 2, response_format="completion",
                                         on_first_content=lambda: calls.append("first"))
            finally:
                server.shutdown()
                thread.join(timeout=2)
        self.assertTrue(report["measurement_valid"], report)
        self.assertEqual(calls, ["first"])

    def test_real_http_fake_service_obeys_release_gate_and_overlaps(self):
        prepared = workload()
        ids = {tuple(r["body"]["prompt"]): r["request_id"] for r in prepared["requests"]}
        all_longs_arrived = threading.Event()
        lock = threading.Lock()
        received_longs = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                request_id = ids[tuple(body["prompt"])]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                if request_id.startswith("long"):
                    with lock:
                        received_longs.append(request_id)
                        if len(received_longs) == 3:
                            all_longs_arrived.set()
                self.send_payload(text_chunk("首"))
                if request_id.startswith("short"):
                    # 用因果事件保持短请求活动，验证客户端确实流式读取并触发长请求。
                    all_longs_arrived.wait(timeout=4)
                self.send_payload(text_chunk("尾", "length"))
                self.send_payload(usage_chunk(len(body["prompt"]), body["max_tokens"]))
                self.send_payload("[DONE]")

            def send_payload(self, payload):
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                result = run_mixed(prepared, f"http://127.0.0.1:{server.server_port}/v1",
                                   socket_timeout_s=5, gate_timeout_s=3)
            finally:
                server.shutdown()
                thread.join(timeout=2)
        self.assertTrue(result["summary"]["pilot_pass"], result)
        self.assertEqual(len(result["first_content_gate"]["seen"]), 4)
        self.assertEqual(len(received_longs), 3)
        self.assertEqual(len(result["requests"]), 7)


if __name__ == "__main__":
    unittest.main()
