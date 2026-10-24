"""人工 SSE 校验采集口径；不启动 vLLM、不把构造时间当性能结果。"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from stage_3_serving.stream_probe import StreamCapture, iter_sse_data, measure_request


def chunk(content=None, finish=None):
    delta = {} if content is None else {"content": content}
    return json.dumps({"choices": [{"index": 0, "delta": delta,
                                    "finish_reason": finish}]}, ensure_ascii=False)


def usage(prompt=2, output=5):
    return json.dumps({"choices": [], "usage": {"prompt_tokens": prompt,
                      "completion_tokens": output, "total_tokens": prompt + output}})


class StreamProbeTests(unittest.TestCase):
    def complete(self, output=5):
        capture = StreamCapture()
        capture.consume(chunk("你好"), 0.1)
        capture.consume(chunk("世界"), 0.5)
        capture.consume(chunk(finish="length"), 0.51)
        capture.consume(usage(output=output), 0.515)
        capture.consume("[DONE]", 0.52)
        return capture

    def test_sse_comments_crlf_and_multiline_framing(self):
        lines = [b": keepalive\r\n", b"\r\n", b"data: first\r\n",
                 b"data: second\r\n", b"\r\n", b"data: [DONE]\n", b"\n"]
        self.assertEqual(list(iter_sse_data(lines)), ["first\nsecond", "[DONE]"])

    def test_unterminated_sse_event_is_not_dispatched(self):
        self.assertEqual(list(iter_sse_data([b"data: [DONE]\n"])), [])

    def test_role_and_empty_body_do_not_start_ttft(self):
        capture = StreamCapture()
        capture.consume(json.dumps({"choices": [{"index": 0,
                        "delta": {"role": "assistant", "content": ""}}]}), 0.01)
        self.assertIsNone(capture.first_content_s)
        capture.consume(chunk("正文"), 0.1)
        self.assertEqual(capture.first_content_s, 0.1)

    def test_times_and_usage_not_chunk_count_define_metrics(self):
        report = self.complete().report()
        self.assertTrue(report["measurement_valid"])
        self.assertEqual(report["content_chunk_count"], 2)
        self.assertEqual(report["usage"]["completion_tokens"], 5)
        self.assertEqual(report["times_s"], {"send": 0.0, "first_content": 0.1,
                                            "last_content": 0.5, "done": 0.52})
        self.assertAlmostEqual(report["metrics"]["client_ttft_ms"], 100)
        self.assertAlmostEqual(report["metrics"]["client_e2e_ms"], 520)
        self.assertAlmostEqual(report["metrics"]["client_tpot_ms"], 105)

    def test_space_is_nonempty_content(self):
        capture = StreamCapture()
        capture.consume(chunk(" "), 0.0)
        capture.consume(chunk("a"), 0.1)
        self.assertEqual(capture.first_content_s, 0.0)
        self.assertEqual("".join(capture.text_parts), " a")

    def test_single_output_tpot_is_missing_not_zero(self):
        report = self.complete(output=1).report()
        self.assertTrue(report["measurement_valid"])
        self.assertIsNone(report["metrics"]["client_tpot_ms"])

    def test_missing_usage_is_invalid_without_guessing_tokens(self):
        capture = StreamCapture()
        capture.consume(chunk("正文", "stop"), 0.1)
        capture.consume("[DONE]", 0.2)
        report = capture.report()
        self.assertFalse(report["measurement_valid"])
        self.assertIsNone(report["usage"])
        self.assertIsNone(report["metrics"]["client_tpot_ms"])
        self.assertIn("missing_or_invalid_usage", report["issues"])

    def test_missing_done_does_not_report_successful_e2e(self):
        capture = StreamCapture()
        capture.consume(chunk("正文", "stop"), 0.1)
        capture.consume(usage(), 0.2)
        report = capture.report()
        self.assertFalse(report["measurement_valid"])
        self.assertIsNone(report["metrics"]["client_e2e_ms"])

    def test_malformed_usage_is_not_accepted(self):
        for invalid in [{"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 6},
                        {"prompt_tokens": 2, "completion_tokens": True, "total_tokens": 3},
                        {"prompt_tokens": 2, "completion_tokens": -1, "total_tokens": 1}]:
            capture = self.complete()
            capture.usage = invalid
            self.assertFalse(capture.report()["measurement_valid"])

    def test_server_error_and_multiple_choices_are_rejected(self):
        for payload in [json.dumps({"error": {"message": "failure"}}),
                        json.dumps({"choices": [{"index": 0}, {"index": 1}]})]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                StreamCapture().consume(payload, 0.1)

    def test_time_order_and_after_done_are_checked(self):
        capture = self.complete()
        with self.assertRaises(ValueError):
            capture.consume(chunk("extra"), 0.6)
        for bad_time in [-1, float("nan"), float("inf")]:
            with self.subTest(time=bad_time), self.assertRaises(ValueError):
                StreamCapture().consume(chunk("text"), bad_time)
        other = StreamCapture()
        other.consume(chunk("a"), 0.2)
        with self.assertRaises(ValueError):
            other.consume(chunk("b"), 0.1)

    def test_transport_error_prevents_valid_measurement(self):
        report = self.complete().report({"type": "TimeoutError", "message": "test"})
        self.assertFalse(report["measurement_valid"])
        self.assertIn("transport_or_protocol_error", report["issues"])
        self.assertIsNone(report["metrics"]["client_tpot_ms"])

    def test_http_path_with_local_fake_sse_server(self):
        # 只验证 HTTP->SSE->结果的真实调用路径；该服务不包含任何推理引擎。
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append((self.path, json.loads(self.rfile.read(
                    int(self.headers["Content-Length"])).decode("utf-8"))))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.end_headers()
                for payload in [chunk("中文"), chunk("正文", "length"), usage(output=4), "[DONE]"]:
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                report = measure_request(
                    f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                    {"model": "fake", "stream": True}, socket_timeout_s=2)
            finally:
                server.shutdown()
                thread.join(timeout=2)
        self.assertTrue(report["measurement_valid"], report)
        self.assertEqual(report["http_status"], 200)
        self.assertEqual(report["text"], "中文正文")
        self.assertEqual(report["usage"]["completion_tokens"], 4)
        self.assertEqual(received, [("/v1/chat/completions", {"model": "fake", "stream": True})])


if __name__ == "__main__":
    unittest.main()
