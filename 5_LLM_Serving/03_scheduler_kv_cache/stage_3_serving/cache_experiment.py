"""5.3.3 剩余两组的辅助采集：容量边界与独立前缀缓存 workload。

不承担服务生命周期，不安装环境。调用已有流式采集器，保留每条请求、
计时边界、指标原文和失败。容量组只比较 N=2/3；前缀组独立输入、独立引擎。
"""

import argparse
import hashlib
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from stage_3_serving.budget_experiment import (
    MODEL_PATH, REVISION, capture_environment, expected_options, metric_samples, read_metrics,
)
from stage_3_serving.prepare_mixed_workload import make_exact_prompt
from stage_3_serving.stream_probe import measure_request


def request_spec(name, prompt, output):
    return {"request_id": name, "expected_prompt_tokens": len(prompt),
            "expected_completion_tokens": output,
            "body": {"model": "qwen2.5-1.5b", "prompt": prompt, "max_tokens": output,
                     "n": 1, "temperature": 0, "ignore_eos": True,
                     "add_special_tokens": False, "stream": True,
                     "stream_options": {"include_usage": True}}}


def build_workload(tokenizer):
    """不同目标在第一个 Token 就分开，保证不共用第一个完整缓存块。"""
    letters = {c: tokenizer.encode(c, add_special_tokens=False) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    if any(len(ids) != 1 for ids in letters.values()) or len({v[0] for v in letters.values()}) != 26:
        raise ValueError("标记必须是 26 个不同的合法单 Token")
    seed = tokenizer.encode("This is a synthetic experiment about memory and scheduling. ", add_special_tokens=False)
    def tagged(letter, length):
        return letters[letter] + make_exact_prompt(seed, length - 1)
    capacity = [request_spec(f"capacity_{i}", tagged(c, 1536), 128)
                for i, c in enumerate("ABC")]
    def prefix(letter):
        return tagged(letter, 1024)
    def query(name, p, suffix):
        return request_spec(name, p + tagged(suffix, 256), 64)
    warmups = [[query(f"warmup_{i}_{j}", prefix(c), s) for j, s in enumerate("LMNO")]
               for i, c in enumerate("YZ")]
    rounds = []
    for i in range(3):
        cold, warm = prefix("ACE"[i]), prefix("BDF"[i])
        rounds.append({"cold": [query(f"cold_{i}", cold, "J")],
                       # 额外 16 个私有位置保证共享的 1024 个位置全部成为已计算整块。
                       "primer": [request_spec(f"primer_{i}", warm + tagged("P", 16), 1)],
                       "warm_single": [query(f"warm_single_{i}", warm, "K")],
                       "warm_batch": [query(f"warm_batch_{i}_{j}", warm, s)
                                      for j, s in enumerate("LMNO")]})
    return {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "revision": REVISION,
            "capacity": capacity, "prefix_warmups": warmups, "prefix_rounds": rounds}


def verify_cache_environment(snapshot, mode, initial=None):
    expected = expected_options("A")
    kv_bytes = 134217728 if mode == "capacity" else 1073741824
    expected["--kv-cache-memory-bytes"] = str(kv_bytes)
    expected["prefix_caching"] = mode == "prefix_on"
    if snapshot["options"] != expected:
        raise ValueError("服务启动参数不符合本组配置")
    cache = snapshot["cache"]
    blocks = kv_bytes // (28672 * 16)
    required = {"block_size": "16", "cache_dtype": "auto", "kv_cache_memory_bytes": str(kv_bytes),
                "enable_prefix_caching": str(mode == "prefix_on"),
                "num_gpu_blocks": str(blocks), "kv_cache_size_tokens": str(blocks * 16)}
    if any(cache.get(k) != v for k, v in required.items()):
        raise ValueError("实际 KV 容量或前缀开关不符合预测")
    if initial is not None:
        for key in ("pid", "start_ticks"):
            if snapshot["process"][key] != initial["process"][key]:
                raise ValueError("实验期间服务发生重启")
        if cache != initial["cache"] or snapshot["packages"] != initial["packages"]:
            raise ValueError("实验期间缓存配置或包版本改变")


def counters(raw):
    names = ("num_preemptions_total", "prefix_cache_queries_total", "prefix_cache_hits_total",
             "generation_tokens_total", "prompt_tokens_total", "request_success_total")
    result = {}
    for name in names:
        samples = metric_samples(raw, "vllm:" + name)
        if not samples:
            raise ValueError("缺少服务计数：" + name)
        result[name] = sum(value for _, value in samples)
    return result


def synchronized_metrics(expected_success):
    """指标可能稍晚更新；在计时窗口外等成功计数追上，不伪造零命中。"""
    deadline = time.monotonic() + 20
    while True:
        raw = read_metrics()
        got = counters(raw)["request_success_total"]
        if got == expected_success:
            return raw
        if got > expected_success:
            raise ValueError("成功请求数超出预期，可能混入其它流量")
        if time.monotonic() >= deadline:
            raise ValueError("服务计数未追上已完成请求，不能计算可靠增量")
        time.sleep(0.1)


def run_batch(specs, monitor=False):
    barrier = threading.Barrier(len(specs))
    stop = threading.Event()
    trace, trace_errors = [], []
    def sampler():
        while not stop.is_set():
            try:
                raw = read_metrics()
                sample = {"perf_counter_s": time.perf_counter()}
                for name in ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc", "num_preemptions_total"):
                    values = metric_samples(raw, "vllm:" + name)
                    if len(values) != 1:
                        raise ValueError("采样必须为单引擎指标")
                    sample[name] = values[0][1]
                trace.append(sample)
            except Exception as exc:
                trace_errors.append(str(exc))
            stop.wait(0.1)
    def send(spec):
        barrier.wait(timeout=10)
        report = measure_request("http://127.0.0.1:8000/v1/completions", spec["body"],
                                 120, response_format="completion")
        report.update({k: spec[k] for k in ("request_id", "expected_prompt_tokens", "expected_completion_tokens")})
        return report
    observer = threading.Thread(target=sampler, daemon=True) if monitor else None
    if observer:
        observer.start()
    try:
        with ThreadPoolExecutor(max_workers=len(specs)) as executor:
            reports = list(executor.map(send, specs))
    finally:
        stop.set()
        if observer:
            observer.join(timeout=12)
    valid = all(r["measurement_valid"] and r["finish_reason"] == "length"
                and r["usage"]["prompt_tokens"] == r["expected_prompt_tokens"]
                and r["usage"]["completion_tokens"] == r["expected_completion_tokens"] for r in reports)
    summary = {"valid": valid}
    if valid:
        start = min(r["t0_perf_counter_s"] for r in reports)
        end = max(r["t0_perf_counter_s"] + r["times_s"]["done"] for r in reports)
        summary.update(http_window_s=end - start,
                       output_tokens_per_s=sum(r["usage"]["completion_tokens"] for r in reports) / (end - start),
                       mean_client_ttft_ms=statistics.mean(r["metrics"]["client_ttft_ms"] for r in reports),
                       mean_client_e2e_ms=statistics.mean(r["metrics"]["client_e2e_ms"] for r in reports))
    return {"requests": reports, "summary": summary, "trace": trace, "trace_errors": trace_errors}


def check_prefix_delta(mode, phase, delta):
    expected = {"cold": 0, "primer": 0, "warm_single": 1024, "warm_batch": 4096}
    if mode == "prefix_off":
        if delta["prefix_cache_queries_total"] != 0 or delta["prefix_cache_hits_total"] != 0:
            raise ValueError("关闭前缀缓存组出现缓存查询或命中")
    elif phase in expected and delta["prefix_cache_hits_total"] != expected[phase]:
        raise ValueError(f"{phase} 命中数不符合隔离后的共享前缀长度：{delta}")
    queries = delta["prefix_cache_queries_total"]
    return delta["prefix_cache_hits_total"] / queries if queries else None


def summarize_phases(phases):
    groups = {}
    for phase in phases:
        if phase["name"].startswith("warmup"):
            continue
        groups.setdefault(phase["name"], []).append(phase)
    summary = {}
    for name, group in groups.items():
        row = {"rounds": len(group)}
        for metric in ("mean_client_ttft_ms", "mean_client_e2e_ms", "output_tokens_per_s"):
            values = [p["batch"]["summary"][metric] for p in group]
            row[metric] = {"values": values, "median": statistics.median(values), "min": min(values), "max": max(values)}
        row["preemptions"] = [p["counter_delta"]["num_preemptions_total"] for p in group]
        row["prefix_hits_tokens"] = [p["counter_delta"]["prefix_cache_hits_total"] for p in group]
        row["prefix_queries_tokens"] = [p["counter_delta"]["prefix_cache_queries_total"] for p in group]
        summary[name] = row
    return summary


def run_experiment(mode, workload):
    result = {"mode": mode, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "valid": False, "error": None, "phases": [], "snapshots": [],
              "scope": "client HTTP window; capacity gauges sampled at 10 Hz, not GPU step trace"}
    try:
        if workload["revision"] != REVISION:
            raise ValueError("模型 revision 不符")
        initial = capture_environment()
        result["snapshots"].append(initial)
        verify_cache_environment(initial, mode)
        success = counters(initial["raw_metrics"])["request_success_total"]
        if success != 0:
            raise ValueError("本组要求刚启动的独立服务，没有其它已完成请求")
        if mode == "capacity":
            jobs = [("warmup_2", i, workload["capacity"][:2]) for i in range(2)]
            jobs += [(f"n{n}", i, workload["capacity"][:n]) for i in range(3) for n in (2, 3)]
        else:
            jobs = [("warmup_prefix", i, specs) for i, specs in enumerate(workload["prefix_warmups"])]
            jobs += [(name, i, block[name]) for i, block in enumerate(workload["prefix_rounds"])
                     for name in ("cold", "primer", "warm_single", "warm_batch")]
        for name, index, specs in jobs:
            before_raw = synchronized_metrics(success)
            phase = {"name": name, "round": index, "before_metrics": before_raw}
            result["phases"].append(phase)
            phase["batch"] = run_batch(specs, monitor=mode == "capacity")
            if not phase["batch"]["summary"]["valid"]:
                raise ValueError("请求协议、完成原因或实际长度验收失败，保留原始记录")
            success += len(specs)
            after_raw = synchronized_metrics(success)
            phase["after_metrics"] = after_raw
            before, after = counters(before_raw), counters(after_raw)
            delta = {k: after[k] - before[k] for k in before}
            if any(v < 0 for v in delta.values()):
                raise ValueError("累计计数回退，不能作为同一引擎的差值")
            phase["counter_delta"] = delta
            if mode != "capacity":
                phase["prefix_hit_rate_token_weighted"] = check_prefix_delta(mode, name, delta)
            snapshot = capture_environment()
            result["snapshots"].append(snapshot)
            verify_cache_environment(snapshot, mode, initial)
            print(f"{mode} {name} round={index + 1}: valid, preemptions={delta['num_preemptions_total']}, prefix_hits={delta['prefix_cache_hits_total']}", flush=True)
        result["summary"] = summarize_phases(result["phases"])
        result["valid"] = True
    except (Exception, KeyboardInterrupt) as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--mode", choices=("capacity", "prefix_off", "prefix_on"), required=True)
    run.add_argument("--workload", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as output:
            if args.command == "prepare":
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=False)
                result = build_workload(tokenizer)
            else:
                data = args.workload.read_bytes()
                result = run_experiment(args.mode, json.loads(data))
                result.update(workload_file=str(args.workload), workload_sha256=hashlib.sha256(data).hexdigest())
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write("\n")
    except FileExistsError:
        parser.error("结果文件已存在，不覆盖；请选择新的运行目录")
    print("saved:", args.output, flush=True)
    if args.command == "run":
        print(json.dumps({"valid": result["valid"], "error": result["error"], "summary": result.get("summary")}, ensure_ascii=False), flush=True)
        return 0 if result["valid"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
