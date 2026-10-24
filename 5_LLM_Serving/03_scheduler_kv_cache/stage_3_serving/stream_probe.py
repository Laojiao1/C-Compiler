"""5.3.3 单请求流式采集辅助工具：先验证口径，不做负载扫描。

在 5.3 代码根目录运行 python -m stage_3_serving.stream_probe。

仅支持 n=1 的文本聊天或 Completion。使用 Python 标准库，不需要安装客户端依赖。
记录客户端完整 SSE 事件的观察时刻，不声称测到了 GPU Token 时间。
"""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


def iter_sse_data(lines):
    """按空行组装完整事件；忽略注释，不把一次网络读取当成一个 Token。"""
    parts = []
    for raw_line in lines:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if parts:
                yield "\n".join(parts)
                parts = []
        elif line.startswith("data:"):
            value = line[5:]
            parts.append(value[1:] if value.startswith(" ") else value)
    # EOF 时不把未结束的事件假装成完整事件；缺少 [DONE] 会令验收失败。


class StreamCapture:
    def __init__(self, response_format="chat"):
        if response_format not in ("chat", "completion"):
            raise ValueError("response_format 必须为 chat 或 completion")
        self.response_format = response_format
        self.events = []
        self.text_parts = []
        self.first_content_s = None
        self.last_content_s = None
        self.done_s = None
        self.finish_reason = None
        self.usage = None

    def consume(self, payload: str, elapsed_s: float) -> None:
        """elapsed_s 来自同一个 perf_counter，表示相对本请求发送起点的秒数。"""
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("事件时间必须是有限的非负数")
        if self.events and elapsed_s < self.events[-1]["elapsed_s"]:
            raise ValueError("事件时间不能倒退")
        if self.done_s is not None:
            raise ValueError("[DONE] 后不应继续处理事件")
        self.events.append({"elapsed_s": elapsed_s, "data": payload})
        if payload == "[DONE]":
            self.done_s = elapsed_s
            return

        chunk = json.loads(payload)
        if not isinstance(chunk, dict) or "error" in chunk:
            raise ValueError("流中收到非响应对象或服务端错误事件")
        if chunk.get("usage") is not None:
            self.usage = chunk["usage"]
        choices = chunk.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError("采集器仅支持 n=1 的响应")
        for choice in choices:
            if not isinstance(choice, dict):
                raise ValueError("choice 必须为对象")
            if choice.get("index", 0) != 0:
                raise ValueError("采集器仅支持 index=0")
            if self.response_format == "completion":
                if "delta" in choice:
                    raise ValueError("Completion 模式收到聊天格式，不能混用接口")
                content = choice.get("text")
            else:
                if "text" in choice and "delta" not in choice:
                    raise ValueError("聊天模式收到 Completion 格式")
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    raise ValueError("delta 必须为对象")
                if delta.get("tool_calls") or delta.get("reasoning_content"):
                    raise ValueError("本工具只验证普通正文，不支持工具或推理分流")
                content = delta.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError("正文必须为字符串")
            if content:
                # 不用 strip：空格本身也可能是有效输出，只有空字符串才忽略。
                if self.first_content_s is None:
                    self.first_content_s = elapsed_s
                self.last_content_s = elapsed_s
                self.text_parts.append(content)
            reason = choice.get("finish_reason")
            if reason is not None:
                self.finish_reason = reason

    def report(self, error=None) -> dict:
        issues = []
        if error is not None:
            issues.append("transport_or_protocol_error")
        if self.done_s is None:
            issues.append("missing_done")
        if self.first_content_s is None:
            issues.append("missing_nonempty_content")
        if self.finish_reason not in ("stop", "length"):
            issues.append("missing_or_unsupported_finish_reason")

        # 严格核对服务端计数。缺失不补猜测值，也不对正文重新分词冒充实测。
        counts = None
        if isinstance(self.usage, dict):
            values = [self.usage.get(key) for key in
                      ("prompt_tokens", "completion_tokens", "total_tokens")]
            if (all(type(value) is int and value >= 0 for value in values)
                    and values[0] + values[1] == values[2]
                    and values[1] > 0):
                counts = values
        if counts is None:
            issues.append("missing_or_invalid_usage")

        valid = not issues
        tpot_ms = None
        if valid and counts[1] > 1:
            # 截止到 [DONE]，因此含末正文后的收尾时间；不是纯 GPU Decode 耗时。
            tpot_ms = (self.done_s - self.first_content_s) * 1000 / (counts[1] - 1)
        return {
            "response_format": self.response_format,
            "measurement_valid": valid,
            "issues": issues,
            "error": error,
            "times_s": {
                "send": 0.0,
                "first_content": self.first_content_s,
                "last_content": self.last_content_s,
                "done": self.done_s,
            },
            "metrics": {
                "client_ttft_ms": None if self.first_content_s is None else self.first_content_s * 1000,
                "client_e2e_ms": None if self.done_s is None else self.done_s * 1000,
                "client_tpot_ms": tpot_ms,
            },
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "saw_done": self.done_s is not None,
            "content_chunk_count": len(self.text_parts),
            "text": "".join(self.text_parts),
            "events": list(self.events),
        }


def measure_request(url: str, body: dict, socket_timeout_s: float = 120, *,
                    response_format="chat", on_first_content=None) -> dict:
    """直连指定服务，不读取系统代理；超时是连接/读取超时，不是整条请求总时限。"""
    if not math.isfinite(socket_timeout_s) or socket_timeout_s <= 0:
        raise ValueError("socket_timeout_s 必须大于 0")
    request = Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                      headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    opener = build_opener(ProxyHandler({}))
    capture = StreamCapture(response_format)
    http_status = None
    headers_s = None
    error = None

    # 请求体准备、JSON 编码均在起点之前；连接、HTTP 和读取均包含在内。
    t0 = time.perf_counter()
    try:
        with opener.open(request, timeout=socket_timeout_s) as response:
            headers_s = time.perf_counter() - t0
            http_status = response.status
            if response.headers.get_content_type() != "text/event-stream":
                raise ValueError("HTTP 响应不是 text/event-stream")
            for payload in iter_sse_data(response):
                had_first_content = capture.first_content_s is not None
                capture.consume(payload, time.perf_counter() - t0)
                # 在首正文已记时之后通知上层；后续正文不再重复触发发送门槛。
                if not had_first_content and capture.first_content_s is not None:
                    if on_first_content is not None:
                        on_first_content()
                if capture.done_s is not None:
                    break
    except (OSError, ValueError, HTTPException) as exc:
        if isinstance(exc, HTTPError):
            http_status = exc.code
        error = {"type": type(exc).__name__, "message": str(exc)}

    report = capture.report(error)
    report.update({
        "url": url, "request": body, "http_status": http_status,
        "clock": "time.perf_counter", "t0_perf_counter_s": t0,
        "response_headers_s": headers_s,
        "observed_until_exit_s": time.perf_counter() - t0,
        "timing_definition": {
            "start": "immediately before HTTP open; after request JSON preparation",
            "content_time": "complete SSE event delivered to parser; nonempty " + (
                "choice.text" if response_format == "completion" else "delta.content"),
            "end": "client observes complete [DONE] SSE event",
            "tpot": "(done - first_content) / (usage.completion_tokens - 1), G > 1",
            "scope": "per-request HTTP interval; caller scheduling delay excluded; not GPU internal timing",
        },
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen2.5-1.5b")
    parser.add_argument("--prompt", default="请用一句话解释大模型推理中 KV Cache 的作用。")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--socket-timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_tokens <= 0 or not math.isfinite(args.socket_timeout) or args.socket_timeout <= 0:
        parser.error("max-tokens 和 socket-timeout 必须大于 0")
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "n": 1, "temperature": 0.0, "max_tokens": args.max_tokens,
        "stream": True, "stream_options": {"include_usage": True},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        # 先排除文件名冲突，不在完成一次请求之后才发现旧结果不能覆盖。
        with args.output.open("x", encoding="utf-8") as output:
            report = measure_request(args.base_url.rstrip("/") + "/chat/completions",
                                     body, args.socket_timeout)
            report["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
            report["purpose"] = "single_request_measurement_smoke_not_performance_comparison"
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write("\n")
    except FileExistsError:
        parser.error("输出文件已存在，请选择新文件名；不会覆盖旧结果")
    print("MEASUREMENT_PASS" if report["measurement_valid"] else "MEASUREMENT_FAILED")
    for name in ("http_status", "finish_reason", "usage", "content_chunk_count", "metrics", "issues", "error"):
        print(f"{name}: {json.dumps(report[name], ensure_ascii=False)}")
    print(f"saved: {args.output}")
    return 0 if report["measurement_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
