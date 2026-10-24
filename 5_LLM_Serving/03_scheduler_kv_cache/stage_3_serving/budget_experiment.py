"""5.3.3 预算/分块对照的辅助入口：核对服务、预热、重复采集、汇总。

只在已有 WSL 服务上采集，不创建或终止引擎。复用 mixed_probe 的事件门控，
不修改输入、计时定义或旧预跑结果。此文件不是学习者需要手写的核心代码。
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

from stage_3_serving.mixed_probe import run_mixed, validate_workload


REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MODEL_PATH = ("/home/laojiao/.cache/huggingface/hub/"
              "models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/" + REVISION)
CONFIGS = {"A": (512, True), "B": (4096, True), "C": (4096, False)}
KV_BYTES = 1073741824
BASE_URL = "http://127.0.0.1:8000/v1"


def expected_options(label):
    budget, chunked = CONFIGS[label]
    return {"model": MODEL_PATH, "--served-model-name": "qwen2.5-1.5b",
            "--host": "127.0.0.1", "--port": "8000", "--dtype": "bfloat16",
            "--tensor-parallel-size": "1", "--max-model-len": "2048",
            "--gpu-memory-utilization": "0.70", "--max-num-seqs": "16",
            "--max-num-batched-tokens": str(budget), "chunked_prefill": chunked,
            "prefix_caching": False, "--generation-config": "vllm",
            "--kv-cache-memory-bytes": str(KV_BYTES), "--block-size": "16"}


def parse_command(argv):
    """只接受本实验启动器的显式参数；未知选项也拒绝，避免隐藏配置混入。"""
    index = argv.index("serve")
    options = {"model": argv[index + 1]}
    flags = {"--enable-chunked-prefill": ("chunked_prefill", True),
             "--no-enable-chunked-prefill": ("chunked_prefill", False),
             "--enable-prefix-caching": ("prefix_caching", True),
             "--no-enable-prefix-caching": ("prefix_caching", False)}
    allowed = set(expected_options("A")) - {"model", "chunked_prefill", "prefix_caching"}
    rest = iter(argv[index + 2:])
    for arg in rest:
        if arg in flags:
            key, value = flags[arg]
        elif arg in allowed:
            key, value = arg, next(rest, None)
            if value is None or value.startswith("--"):
                raise ValueError(f"参数缺少值：{arg}")
        else:
            raise ValueError(f"存在本轮未核实的启动参数：{arg}")
        if key in options:
            raise ValueError(f"重复启动参数：{key}")
        options[key] = value
    return options


def read_engine_process():
    """WSL 本机只读 /proc，不依赖未开启的 /server_info，也不读取进程环境。"""
    if not Path("/proc").is_dir():
        raise RuntimeError("请在 WSL 的 vllm 环境运行正式采集")
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = [part.decode() for part in (entry / "cmdline").read_bytes().split(b"\0") if part]
            # 只识别真实 CLI 进程，不把 shell 命令字符串或 EngineCore 子进程算进去。
            if not any(Path(argv[i - 1]).name == "vllm" and argv[i] == "serve"
                       for i in range(1, min(len(argv), 4))):
                continue
            stat = (entry / "stat").read_text()
            start_ticks = stat[stat.rfind(")") + 2:].split()[19]
            found.append({"pid": int(entry.name), "start_ticks": start_ticks, "argv": argv})
        except (OSError, UnicodeError):
            continue
    if len(found) != 1:
        raise ValueError(f"应只有一个可核实的 vllm serve 进程，实际找到 {len(found)} 个")
    return found[0]


def metric_samples(text, name):
    """读取指定指标；标签用 JSON 转义解码，不把全站其它指标掺进统计。"""
    pattern = re.compile(re.escape(name) + r'(?:\{(.*)\})?\s+([^\s]+)(?:\s+\d+)?$')
    samples = []
    for line in text.splitlines():
        match = pattern.fullmatch(line)
        if match:
            labels = {key: json.loads('"' + value + '"') for key, value in
                      re.findall(r'(\w+)="((?:\\.|[^"\\])*)"', match[1] or "")}
            value = float(match[2])
            if not math.isfinite(value):
                raise ValueError(f"指标不是有限数：{name}")
            samples.append((labels, value))
    return samples


def read_metrics():
    # 与已有流式客户端一致：仅本次本机请求绕过代理，不改系统代理设置。
    with build_opener(ProxyHandler({})).open("http://127.0.0.1:8000/metrics", timeout=10) as response:
        return response.read().decode("utf-8")


def capture_environment():
    """在计时窗口外核对引擎、等待空闲，并保存原始指标与包版本。"""
    process = read_engine_process()
    deadline = time.monotonic() + 10
    while True:
        raw = read_metrics()
        states = [metric_samples(raw, "vllm:" + name)
                  for name in ("num_requests_running", "num_requests_waiting")]
        if any(len(values) != 1 for values in states):
            raise ValueError("缺少单引擎 running/waiting 指标，停止采集")
        if all(values[0][1] == 0 for values in states):
            break
        if time.monotonic() >= deadline:
            raise ValueError("服务仍有其它运行/等待请求，停止采集；不会替你终止请求")
        time.sleep(0.25)
    cache = metric_samples(raw, "vllm:cache_config_info")
    if len(cache) != 1 or cache[0][1] != 1:
        raise ValueError("无法唯一核实实际 KV 配置")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True,
                         check=True, timeout=10).stdout.strip()
    return {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "process": process, "options": parse_command(process["argv"]),
            "cache": cache[0][0], "raw_metrics": raw, "gpu": gpu,
            "packages": {name: importlib.metadata.version(name)
                         for name in ("vllm", "torch", "triton")}}


def verify_environment(snapshot, label, previous=None):
    expected = expected_options(label)
    actual = snapshot["options"]
    differences = {key: {"expected": value, "actual": actual.get(key)}
                   for key, value in expected.items() if actual.get(key) != value}
    if differences or set(actual) != set(expected):
        raise ValueError("服务不是目标配置：" + json.dumps(differences, ensure_ascii=False))
    cache = snapshot["cache"]
    required = {"block_size": "16", "enable_prefix_caching": "False",
                "kv_cache_memory_bytes": str(KV_BYTES), "cache_dtype": "auto"}
    if any(cache.get(key) != value for key, value in required.items()):
        raise ValueError("服务实际 KV 配置与固定预算不一致")
    # 全注意力 BF16 的每个物理块 = 28 层 × K/V × 2 头 × 128 维 × 2 字节 × 16。
    blocks = KV_BYTES // (28672 * 16)
    if int(cache.get("num_gpu_blocks", -1)) != blocks or int(cache.get("kv_cache_size_tokens", -1)) != blocks * 16:
        raise ValueError("实际 KV 容量不符合预测，先查原因，不继续性能对照")
    if previous is not None:
        identity = lambda snap: (snap["process"]["pid"], snap["process"]["start_ticks"])
        if identity(snapshot) != identity(previous) or snapshot["cache"] != previous["cache"]:
            raise ValueError("采集期间引擎重启或 KV 配置变化")


def summarize_rounds(rounds):
    """只汇总全部通过的三轮；保留每轮值，不用三个样本构造 p99。"""
    if len(rounds) != 3 or not all(r["summary"]["pilot_pass"] for r in rounds):
        return {"valid": False, "reason": "要求三轮全部完成长度、流与重叠验收"}
    getters = {
        "output_tokens_per_s": lambda s: s["output_tokens_per_s"],
        "short_mean_client_tpot_ms": lambda s: s["groups"]["short"]["mean_client_tpot_ms"],
        "long_mean_client_ttft_ms": lambda s: s["groups"]["long"]["mean_client_ttft_ms"],
    }
    summary = {"valid": True, "measured_rounds": 3}
    for key, getter in getters.items():
        values = [getter(r["summary"]) for r in rounds]
        summary[key] = {"round_values": values, "median": statistics.median(values),
                        "min": min(values), "max": max(values)}
    return summary


def run_experiment(label, workload):
    result = {"purpose": "controlled_budget_chunked_prefill", "config": label,
              "warmup_rounds": [], "measured_rounds": [], "snapshots": [],
              "summary": {"valid": False}, "error": None,
              "scope": "client-observed finite batch; not GPU step timing or steady-state limit"}
    try:
        validate_workload(workload)
        if workload.get("revision") != REVISION or workload.get("model_id") != "Qwen/Qwen2.5-1.5B-Instruct":
            raise ValueError("工作量不是已固定的模型与 revision")
        initial = capture_environment()
        result["snapshots"].append(initial)
        verify_environment(initial, label)
        # 每轮完整等待所有流结束，再进入下一轮；不会把预热混入正式样本。
        for index in range(5):
            report = run_mixed(workload, BASE_URL)
            report["purpose"] = "warmup" if index < 2 else "measured_controlled_round"
            target = result["warmup_rounds"] if index < 2 else result["measured_rounds"]
            target.append(report)
            if not report["summary"]["pilot_pass"]:
                raise ValueError("本轮长度、流或客户端重叠验收失败；失败记录已保留")
            snapshot = capture_environment()
            result["snapshots"].append(snapshot)
            verify_environment(snapshot, label, initial)
        result["summary"] = summarize_rounds(result["measured_rounds"])
    except (Exception, KeyboardInterrupt) as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return result


def compare_experiments(results):
    if len(results) != 3 or {r["config"] for r in results} != set(CONFIGS):
        raise ValueError("需要 A、B、C 各一份结果")
    if not all(r["summary"].get("valid") and r.get("error") is None for r in results):
        raise ValueError("存在未完成或无效实验，不能挑选成功轮次比较")
    first = results[0]
    for result in results:
        if result["workload_sha256"] != first["workload_sha256"]:
            raise ValueError("工作量文件不同，不能进行本轮对照")
        for snapshot in result["snapshots"]:
            verify_environment(snapshot, result["config"], result["snapshots"][0])
        baseline = first["snapshots"][0]
        if (result["snapshots"][0]["cache"] != baseline["cache"]
                or result["snapshots"][0]["packages"] != baseline["packages"]):
            raise ValueError("配置间实际 KV 容量或软件版本不同")
    by_config = {r["config"]: r for r in results}
    rows = []
    for left, right in (("A", "B"), ("B", "C")):
        for metric in ("long_mean_client_ttft_ms", "short_mean_client_tpot_ms", "output_tokens_per_s"):
            a, b = (by_config[label]["summary"][metric]["median"] for label in (left, right))
            rows.append({"comparison": left + "/" + right, "metric": metric,
                         left: a, right: b, "right_vs_left_percent": (b / a - 1) * 100})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="在指定配置的已有服务上预热 2 轮、测量 3 轮")
    run.add_argument("--config", choices=CONFIGS, required=True)
    run.add_argument("--workload", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare", help="核对三份结果，只输出 A/B 与 B/C")
    compare.add_argument("results", nargs=3, type=Path)
    args = parser.parse_args()
    if args.command == "compare":
        try:
            rows = compare_experiments([json.loads(path.read_text(encoding="utf-8")) for path in args.results])
        except (OSError, ValueError, KeyError) as exc:
            parser.error(str(exc))
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
        return 0
    workload_bytes = args.workload.read_bytes()
    workload = json.loads(workload_bytes)
    validate_workload(workload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        # 先独占创建输出，再进行任何推理；防止实验做完才发现会覆盖旧文件。
        with args.output.open("x", encoding="utf-8") as output:
            result = run_experiment(args.config, workload)
            result.update(workload_file=str(args.workload),
                          workload_sha256=hashlib.sha256(workload_bytes).hexdigest())
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write("\n")
    except FileExistsError:
        parser.error("输出文件已存在，请换新名称，不覆盖旧结果")
    print("BUDGET_EXPERIMENT_PASS" if result["summary"]["valid"] else "BUDGET_EXPERIMENT_FAILED")
    print("summary:", json.dumps(result["summary"], ensure_ascii=False))
    print("error:", json.dumps(result["error"], ensure_ascii=False))
    print("saved:", args.output)
    return 0 if result["summary"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
