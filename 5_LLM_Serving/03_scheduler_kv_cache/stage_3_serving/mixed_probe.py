"""5.3.3 一轮混合负载试运行：固定输入，短请求首正文齐备后再发长请求。

不启动引擎、不改服务配置、不循环扫描。本轮只验收请求长度和客户端重叠。
完整保留失败请求；不把有限批次吞吐称为硬件极限或稳定服务能力。
"""

import argparse
import json
import math
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from stage_3_serving.stream_probe import StreamCapture, measure_request


def validate_workload(workload):
    """发送前检查工作量，不允许输入被改成另一套实验还沿用原标签。"""
    if workload.get("api_path") != "/v1/completions":
        raise ValueError("本轮只支持 /v1/completions")
    if workload.get("requirements", {}).get("enable_prefix_caching") is not False:
        raise ValueError("普通调度基线要求关闭 prefix caching")
    requests = workload.get("requests", [])
    ids = [request["request_id"] for request in requests]
    if len(ids) != len(set(ids)):
        raise ValueError("请求 ID 重复")
    expected_counts = {"short": 4, "long": 3}
    for cohort, count in expected_counts.items():
        group = [request for request in requests if request["cohort"] == cohort]
        if len(group) != count:
            raise ValueError("本轮要求 4 条短请求和 3 条长请求")
        isl, osl = (128, 256) if cohort == "short" else (1536, 32)
        for request in group:
            body = request["body"]
            if (request["expected_prompt_tokens"] != isl
                    or request["expected_completion_tokens"] != osl
                    or not isinstance(body.get("prompt"), list)
                    or len(body["prompt"]) != isl
                    or any(type(token) is not int or token < 0 for token in body["prompt"])
                    or body.get("max_tokens") != osl
                    or body.get("n") != 1 or body.get("temperature") != 0
                    or body.get("ignore_eos") is not True
                    or body.get("add_special_tokens") is not False
                    or body.get("stream") is not True
                    or body.get("stream_options", {}).get("include_usage") is not True):
                raise ValueError(f"{request['request_id']} 的长度或采样参数与既定工作量不符")
    if len(requests) != 7:
        raise ValueError("存在未识别的请求类别")
    if len({request["body"]["model"] for request in requests}) != 1:
        raise ValueError("本轮请求必须使用同一个服务模型名")


def summarize(reports, gate_opened):
    """按每条请求自己的 t0 对齐到同一个进程时钟，再检查观察窗口交集。"""
    by_id = {report["request_id"]: report for report in reports}
    groups = {name: [r for r in reports if r["cohort"] == name]
              for name in ("short", "long")}
    for report in reports:
        usage = report.get("usage") or {}
        report["lengths_match"] = (
            usage.get("prompt_tokens") == report["expected_prompt_tokens"]
            and usage.get("completion_tokens") == report["expected_completion_tokens"])

    complete = all(r.get("measurement_valid") and r["lengths_match"] for r in reports)
    pairs = []
    ordered = gate_opened
    for long in groups["long"]:
        long_start = long.get("t0_perf_counter_s")
        long_first = long.get("times_s", {}).get("first_content")
        if long_start is None or long_first is None:
            ordered = False
            continue
        for short in groups["short"]:
            short_start = short.get("t0_perf_counter_s")
            short_first = short.get("times_s", {}).get("first_content")
            short_last = short.get("times_s", {}).get("last_content")
            if short_start is None or short_first is None or short_last is None:
                ordered = False
                continue
            if long_start < short_start + short_first:
                ordered = False
            # 使用末正文而非 [DONE]，避免只重叠了短请求的收尾阶段。
            overlap_s = min(long_start + long_first, short_start + short_last) - max(
                long_start, short_start + short_first)
            if overlap_s > 0:
                pairs.append({"long_id": long["request_id"], "short_id": short["request_id"],
                              "overlap_ms": overlap_s * 1000})
    overlap = bool(groups["long"]) and all(
        any(pair["long_id"] == long["request_id"] for pair in pairs)
        for long in groups["long"])

    duration_s = None
    throughput = None
    if complete:
        start = min(r["t0_perf_counter_s"] for r in reports)
        end = max(r["t0_perf_counter_s"] + r["times_s"]["done"] for r in reports)
        duration_s = end - start
        if duration_s > 0:
            throughput = sum(r["usage"]["completion_tokens"] for r in reports) / duration_s

    group_metrics = {}
    for name, group in groups.items():
        valid = [r for r in group if r.get("measurement_valid") and r["lengths_match"]]
        group_metrics[name] = {"total": len(group), "valid": len(valid)}
        for key in ("client_ttft_ms", "client_tpot_ms"):
            values = [r["metrics"][key] for r in valid if r["metrics"].get(key) is not None]
            group_metrics[name]["mean_" + key] = (
                statistics.mean(values) if len(values) == len(group) and values else None)
    return {
        "pilot_pass": complete and ordered and overlap,
        "all_requests_valid_and_exact_length": complete,
        "all_short_first_content_before_long_sends": ordered,
        "each_long_client_ttft_window_overlaps_short_content": overlap,
        "client_overlap_pairs": pairs,
        "batch_http_window_s": duration_s,
        "output_tokens_per_s": throughput,
        "groups": group_metrics,
        "invalid_request_ids": [name for name, r in by_id.items()
                                if not r.get("measurement_valid") or not r["lengths_match"]],
        "scope": "client-observed overlap only; not proof of GPU step co-batching",
    }


def run_mixed(workload, base_url="http://127.0.0.1:8000/v1", *,
              socket_timeout_s=120, gate_timeout_s=30):
    validate_workload(workload)
    if any(not math.isfinite(t) or t <= 0 for t in (socket_timeout_s, gate_timeout_s)):
        raise ValueError("超时必须为有限正数")
    requests = workload["requests"]
    shorts = [r for r in requests if r["cohort"] == "short"]
    longs = [r for r in requests if r["cohort"] == "long"]
    condition = threading.Condition()
    first_seen = set()
    completed_short = set()
    submitted = {}

    def run_one(spec):
        def notify_first():
            if spec["cohort"] == "short":
                with condition:
                    first_seen.add(spec["request_id"])
                    condition.notify_all()

        try:
            report = measure_request(base_url.rstrip("/") + "/completions", spec["body"],
                                     socket_timeout_s, response_format="completion",
                                     on_first_content=notify_first)
        except Exception as exc:
            # 辅助客户端自身异常也必须落盘，不能丢掉慢请求或失败请求。
            report = StreamCapture("completion").report(
                {"type": type(exc).__name__, "message": str(exc)})
        finally:
            if spec["cohort"] == "short":
                with condition:
                    completed_short.add(spec["request_id"])
                    condition.notify_all()
        report.update({key: spec[key] for key in
                       ("request_id", "cohort", "expected_prompt_tokens", "expected_completion_tokens")})
        report["client_task_submitted_perf_s"] = submitted[spec["request_id"]]
        report["client_dispatch_delay_s"] = (report["t0_perf_counter_s"] - submitted[spec["request_id"]]
                                              if "t0_perf_counter_s" in report else None)
        return report

    futures = {}
    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        for spec in shorts:
            submitted[spec["request_id"]] = time.perf_counter()
            futures[spec["request_id"]] = executor.submit(run_one, spec)
        with condition:
            condition.wait_for(lambda: len(first_seen) == len(shorts)
                               or bool(completed_short - first_seen), timeout=gate_timeout_s)
            gate_opened = len(first_seen) == len(shorts)
            gate_seen = sorted(first_seen)
        gate_release_s = time.perf_counter() if gate_opened else None
        if gate_opened:
            for spec in longs:
                submitted[spec["request_id"]] = time.perf_counter()
                futures[spec["request_id"]] = executor.submit(run_one, spec)
        reports = []
        for spec in requests:
            if spec["request_id"] in futures:
                reports.append(futures[spec["request_id"]].result())
            else:
                reports.append({key: spec[key] for key in
                                ("request_id", "cohort", "expected_prompt_tokens", "expected_completion_tokens")} |
                               {"measurement_valid": False, "not_sent": True,
                                "issues": ["short_first_content_gate_not_reached"]})
    return {
        "purpose": "one_round_overlap_pilot_not_controlled_performance_comparison",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": workload.get("model_id"), "revision": workload.get("revision"),
        "engine_configuration": "existing service; not independently verified by this client",
        "first_content_gate": {"opened": gate_opened, "seen": gate_seen,
                               "release_perf_s": gate_release_s, "timeout_s": gate_timeout_s},
        "summary": summarize(reports, gate_opened), "requests": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--socket-timeout", type=float, default=120)
    parser.add_argument("--gate-timeout", type=float, default=30)
    args = parser.parse_args()
    workload = json.loads(args.workload.read_text(encoding="utf-8"))
    validate_workload(workload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as output:
            result = run_mixed(workload, args.base_url, socket_timeout_s=args.socket_timeout,
                               gate_timeout_s=args.gate_timeout)
            result["workload_file"] = str(args.workload)
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write("\n")
    except FileExistsError:
        parser.error("输出文件已存在，请选择新文件名；不会覆盖旧结果")
    print("MIXED_PILOT_PASS" if result["summary"]["pilot_pass"] else "MIXED_PILOT_FAILED")
    for report in result["requests"]:
        usage = report.get("usage") or {}
        print(f"{report['request_id']}: valid={report.get('measurement_valid')}, "
              f"ISL={usage.get('prompt_tokens')}, OSL={usage.get('completion_tokens')}, "
              f"TTFT_ms={report.get('metrics', {}).get('client_ttft_ms')}, "
              f"TPOT_ms={report.get('metrics', {}).get('client_tpot_ms')}")
    print("summary:", json.dumps(result["summary"], ensure_ascii=False))
    print("saved:", args.output)
    return 0 if result["summary"]["pilot_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
