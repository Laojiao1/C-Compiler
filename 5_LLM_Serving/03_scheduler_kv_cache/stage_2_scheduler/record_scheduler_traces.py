"""输出复盘用的 CPU 调度轨迹；只调用已有接口，不修改核心调度策略。

用法：在 5.3 代码根目录运行 python -m stage_2_scheduler.record_scheduler_traces --case all
输出为 JSON，可由终端保存。step 是离散编号，不记录或伪造 GPU 延迟。
"""

import argparse
import json
import platform
from dataclasses import asdict

import torch

from stage_1_paged_kv.paged_kv import BlockPool, RequestKV
from stage_1_paged_kv.paged_kv_store import PagedKVStore
from stage_2_scheduler.scheduler import SchedulingRequest
from stage_2_scheduler.teaching_scheduler import TeachingScheduler


def make_scheduler(blocks, budget, seqs):
    store = PagedKVStore(BlockPool(blocks), 4, 1, 2)
    return TeachingScheduler(store, budget, seqs, 32)


def make_request(scheduler, name, prompt, limit, computed=0, generated=0):
    kv = RequestKV(scheduler.store.pool, 4)
    values = torch.zeros(computed, 1, 2, dtype=torch.float32)
    # 为预先准备好的 Decode 请求真正写入占位 KV，而不是仅修改有效长度。
    scheduler.store.append(kv, values, values)
    return SchedulingRequest(name, prompt, kv, generated, limit)


def snapshot_step(scheduler):
    trace = scheduler.step()
    trace["plan"] = [asdict(chunk) for chunk in trace["plan"]]
    trace["phase_tokens"] = {
        phase: sum(chunk["num_tokens"] for chunk in trace["plan"]
                   if chunk["phase"] == phase)
        for phase in ("prefill", "decode", "recompute")
    }
    trace["generated_tokens"] = {
        name: request.num_generated_tokens for name, request in scheduler.requests.items()
    }
    trace["recompute_until"] = {
        name: request.recompute_until for name, request in scheduler.requests.items()
    }
    return trace


def finish_with_limit(scheduler, trace, max_steps=20):
    # 测试策略可能停滞，因此辅助记录程序也必须有上限，不能无限循环。
    while not all(request.finished for request in scheduler.requests.values()):
        if scheduler.step_index >= max_steps:
            raise RuntimeError("达到轨迹步数上限，仍有未完成请求")
        trace.append(snapshot_step(scheduler))


def run_mixed():
    scheduler = make_scheduler(blocks=32, budget=8, seqs=3)
    for request, arrival in [
        (make_request(scheduler, "C", 10, 3), 0),
        (make_request(scheduler, "A", 8, 4, computed=8, generated=1), 0),
        (make_request(scheduler, "B", 8, 4, computed=8, generated=1), 0),
        (make_request(scheduler, "D", 3, 2), 3),
    ]:
        scheduler.submit(request, arrival_step=arrival)
    trace = []
    finish_with_limit(scheduler, trace)
    return {
        "config": {"num_blocks": 32, "block_size": 4, "token_budget": 8,
                   "max_num_seqs": 3, "max_model_len": 32},
        "initial_generated_tokens": {"A": 1, "B": 1, "C": 0, "D": 0},
        "trace": trace,
    }


def run_preemption():
    scheduler = make_scheduler(blocks=4, budget=14, seqs=2)
    for name in ("A", "B"):
        scheduler.submit(make_request(scheduler, name, 7, 4))
    trace = [snapshot_step(scheduler) for _ in range(3)]
    # 先观察空计划，再由调用者显式抢占 B；这里没有自动牺牲者选择器。
    if trace[-1]["plan"] or trace[-1]["blocked"] != ("A", "B"):
        raise AssertionError("预期的四块池停滞没有出现")
    event = scheduler.preempt("B")
    finish_with_limit(scheduler, trace)
    return {
        "config": {"num_blocks": 4, "block_size": 4, "token_budget": 14,
                   "max_num_seqs": 2, "max_model_len": 32},
        "initial_generated_tokens": {"A": 0, "B": 0},
        "explicit_preemption": event,
        "trace": trace,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("mixed", "preemption", "all"), default="all")
    args = parser.parse_args()
    runners = {"mixed": run_mixed, "preemption": run_preemption}
    chosen = runners if args.case == "all" else {args.case: runners[args.case]}
    result = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "device": "cpu", "dtype": "float32", "num_layers": 1,
            "num_kv_heads": 1, "head_dim": 2,
            "real_model_loaded": False,
        },
        "cases": {name: runner() for name, runner in chosen.items()},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
