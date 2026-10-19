"""比较 GPT-2 原生 Prefill Attention 与 OperatorEngine Prefill Attention。

只替换 Prefill 中的 Attention；生成阶段的 Decode Attention 保持 PyTorch 实现。
接入时产生的 Q/K/V contiguous 拷贝也包含在测量里。
"""

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch

# 脚本位于 GPT-2 项目的 benchmarks/ 子目录；OperatorEngine 是同一仓库中的独立包。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OPERATOR_ENGINE_ROOT = PROJECT_ROOT / "OperatorEngine"
for import_root in (PROJECT_ROOT, OPERATOR_ENGINE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import my_ops
import my_ops.functional as operator_engine_functional
from benchmark_sweep import _load_model, _make_input_ids, _parse_int_list, _run_prealloc, _sync
from decode import AttentionImpl, decode_with_cache


def _operator_engine_prefill_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    """走 OperatorEngine 统一入口，让它按当前规则选择实际 provider。"""
    return my_ops.attention(
        query,
        key,
        value,
        causal=causal,
        provider="auto",
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, int((percentile / 100.0) * len(ordered) + 0.999999) - 1)
    return ordered[min(index, len(ordered) - 1)]


@torch.no_grad()
def _check_numerics(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    output_tokens: int,
) -> Dict[str, float]:
    """只比较共享上下文下的 logits，并报告生成序列何时开始分叉。"""
    baseline_ids, baseline_logits = decode_with_cache(
        model,
        input_ids,
        output_tokens,
        mode="prealloc",
    )
    operator_ids, operator_logits = decode_with_cache(
        model,
        input_ids,
        output_tokens,
        mode="prealloc",
        prefill_attention_impl=_operator_engine_prefill_attention,
    )

    if len(baseline_logits) != len(operator_logits):
        raise AssertionError("baseline 和 OperatorEngine 产生的 logits 步数不一致")

    # 某一步的 logits 会决定同一步生成的 token。若 token 不同，后续两条路径
    # 看到的历史就不同，后续 logits 不能再拿来判断 Attention 算子是否对齐。
    # 所以每一步只比较此前生成历史完全相同的 batch 行。
    active_rows = torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    total_abs_error = 0.0
    compared_values = 0
    max_abs_error = 0.0
    common_prefix_lengths = [0] * input_ids.shape[0]
    first_divergence = None

    for step, (reference, actual) in enumerate(zip(baseline_logits, operator_logits)):
        if active_rows.any():
            reference_shared = reference[active_rows]
            actual_shared = actual[active_rows]
            difference = (reference_shared - actual_shared).abs()

            # 按原始 logits 精度统计误差；fp16 数值本身有舍入，允许合理的误差范围。
            max_abs_error = max(max_abs_error, difference.max().item())
            total_abs_error += difference.sum().item()
            compared_values += difference.numel()
            torch.testing.assert_close(
                actual_shared,
                reference_shared,
                rtol=2e-2,
                atol=2e-2,
                msg=f"共享上下文下第 {step + 1} 步 logits 未对齐",
            )

            # 当前 logits 预测的 token 已经生成在 sequence 的 prompt_len + step 处。
            # 记录每个样本共同前缀长度，并决定下一步哪些样本仍可继续比较。
            same_token = baseline_ids[:, input_ids.shape[1] + step].eq(
                operator_ids[:, input_ids.shape[1] + step]
            )
            for row in range(input_ids.shape[0]):
                if active_rows[row] and same_token[row]:
                    common_prefix_lengths[row] += 1
                elif active_rows[row] and first_divergence is None:
                    # 贪心选择只看最大值。这里记录前两名的差距，帮助判断
                    # 分叉是否发生在两个候选 token 几乎打平的时候。
                    baseline_top2 = torch.topk(reference[row], k=2).values
                    operator_top2 = torch.topk(actual[row], k=2).values
                    first_divergence = {
                        "step": step + 1,
                        "row": row,
                        "baseline_token": int(baseline_ids[row, input_ids.shape[1] + step]),
                        "operator_token": int(operator_ids[row, input_ids.shape[1] + step]),
                        "baseline_margin": (baseline_top2[0] - baseline_top2[1]).item(),
                        "operator_margin": (operator_top2[0] - operator_top2[1]).item(),
                    }
            active_rows &= same_token

    result = {
        "max_abs_error": max_abs_error,
        "mean_abs_error": total_abs_error / compared_values if compared_values else 0.0,
        "tokens_match": float(torch.equal(baseline_ids, operator_ids)),
        "common_prefix_min": float(min(common_prefix_lengths, default=0)),
        "common_prefix_mean": statistics.mean(common_prefix_lengths) if common_prefix_lengths else 0.0,
        "first_divergence_step": float(first_divergence["step"]) if first_divergence else 0.0,
        "first_divergence_baseline_token": float(first_divergence["baseline_token"]) if first_divergence else -1.0,
        "first_divergence_operator_token": float(first_divergence["operator_token"]) if first_divergence else -1.0,
        "first_divergence_baseline_margin": first_divergence["baseline_margin"] if first_divergence else 0.0,
        "first_divergence_operator_margin": first_divergence["operator_margin"] if first_divergence else 0.0,
    }

    # 结果张量只用于正确性检查，先释放它们，避免污染后续的峰值显存测量。
    del baseline_ids, baseline_logits, operator_ids, operator_logits
    gc.collect()
    _sync(input_ids.device)
    return result


def _measure_variant(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    output_tokens: int,
    attention_impl: AttentionImpl | None,
    warmup_runs: int,
    measured_runs: int,
) -> Dict[str, float]:
    """成对计时一条路径；峰值增量以模型和输入已驻留为 baseline。"""
    device = input_ids.device
    for _ in range(warmup_runs):
        _run_prealloc(model, input_ids, output_tokens, attention_impl)

    latency_samples: List[float] = []
    peak_extra_samples: List[float] = []
    peak_reserved_samples: List[float] = []

    for _ in range(measured_runs):
        _sync(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()
        _run_prealloc(model, input_ids, output_tokens, attention_impl)
        _sync(device)
        latency_samples.append((time.perf_counter() - start) * 1000.0)
        peak_extra_samples.append(
            (torch.cuda.max_memory_allocated(device) - baseline_allocated) / (1024 ** 2)
        )
        peak_reserved_samples.append(torch.cuda.max_memory_reserved(device) / (1024 ** 2))

    return {
        "latency_p50_ms": statistics.median(latency_samples),
        "latency_p95_ms": _percentile(latency_samples, 95),
        "peak_extra_mib": statistics.median(peak_extra_samples),
        "peak_reserved_mib": max(peak_reserved_samples),
    }


def _run_case(
    model: torch.nn.Module,
    batch_size: int,
    prompt_len: int,
    output_tokens: int,
    warmup_runs: int,
    measured_runs: int,
    device: torch.device,
) -> None:
    input_ids = _make_input_ids(batch_size, prompt_len, device)
    numeric = _check_numerics(model, input_ids, output_tokens)

    selected_provider = operator_engine_functional._select_attention_provider(
        prompt_len,
        causal=True,
    )
    if selected_provider != "triton":
        raise RuntimeError(f"本次 causal Prefill 预期进入 Triton，实际为 {selected_provider}")

    # 两条路径都先跑 warmup，避免首次 kernel 初始化混入测量。
    for _ in range(warmup_runs):
        _run_prealloc(model, input_ids, output_tokens, None)
        _run_prealloc(
            model,
            input_ids,
            output_tokens,
            _operator_engine_prefill_attention,
        )

    # 两条路径交替先后顺序，减少 GPU 时钟升降对单一方案的偏袒。
    samples: Dict[str, List[Dict[str, float]]] = {"torch": [], "operator_engine": []}
    for round_index in range(measured_runs):
        variants = (
            [("torch", None), ("operator_engine", _operator_engine_prefill_attention)]
            if round_index % 2 == 0
            else [("operator_engine", _operator_engine_prefill_attention), ("torch", None)]
        )
        for name, attention_impl in variants:
            samples[name].append(
                _measure_variant(
                    model,
                    input_ids,
                    output_tokens,
                    attention_impl,
                    warmup_runs=0,
                    measured_runs=1,
                )
            )

    results = {
        name: {
            key: statistics.median(sample[key] for sample in variant_samples)
            for key in variant_samples[0]
        }
        for name, variant_samples in samples.items()
    }

    baseline = results["torch"]
    candidate = results["operator_engine"]
    speedup = baseline["latency_p50_ms"] / candidate["latency_p50_ms"]
    baseline_tps = batch_size * output_tokens * 1000.0 / baseline["latency_p50_ms"]
    candidate_tps = batch_size * output_tokens * 1000.0 / candidate["latency_p50_ms"]

    print(
        f"B={batch_size}, prompt={prompt_len}, output={output_tokens} | "
        f"Q/K/V=[{batch_size},12,{prompt_len},64], causal=True, "
        f"auto provider={selected_provider} | "
        f"shared-context max/mean logit abs err="
        f"{numeric['max_abs_error']:.6g}/{numeric['mean_abs_error']:.6g}, "
        f"tokens match={bool(numeric['tokens_match'])}, "
        f"common output prefix min/mean="
        f"{numeric['common_prefix_min']:.0f}/{numeric['common_prefix_mean']:.2f}"
    )
    if numeric["first_divergence_step"]:
        print(
            f"  first token divergence: generated token {numeric['first_divergence_step']:.0f}, "
            f"Torch/OE id={numeric['first_divergence_baseline_token']:.0f}/"
            f"{numeric['first_divergence_operator_token']:.0f}, "
            f"top1-top2 margin={numeric['first_divergence_baseline_margin']:.6g}/"
            f"{numeric['first_divergence_operator_margin']:.6g}"
        )
    print(
        f"  Torch: {baseline['latency_p50_ms']:.3f} ms p50, "
        f"{baseline_tps:.2f} tokens/s, peak extra {baseline['peak_extra_mib']:.2f} MiB"
    )
    print(
        f"  OE:    {candidate['latency_p50_ms']:.3f} ms p50, "
        f"{candidate_tps:.2f} tokens/s, peak extra {candidate['peak_extra_mib']:.2f} MiB, "
        f"speedup {speedup:.3f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=_parse_int_list, default=[1])
    parser.add_argument("--prompt-lengths", type=_parse_int_list, default=[32, 128])
    parser.add_argument("--output-lengths", type=_parse_int_list, default=[8])
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--measured-runs", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要 CUDA，请在已配置 GPU 的 WSL 环境运行。")
    if args.warmup_runs < 0 or args.measured_runs <= 0:
        raise ValueError("warmup-runs 不能小于 0，measured-runs 必须大于 0。")

    device = torch.device("cuda")
    model = _load_model(torch.float16, device)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("dtype: fp16; cache: preallocated; only Prefill Attention is replaced")
    print("Q/K/V contiguous copies are inside the timed region; Decode stays on PyTorch.")
    print(f"warmup per variant: {args.warmup_runs}; paired measured rounds: {args.measured_runs}")

    for batch_size in args.batch_sizes:
        for prompt_len in args.prompt_lengths:
            for output_tokens in args.output_lengths:
                _run_case(
                    model,
                    batch_size,
                    prompt_len,
                    output_tokens,
                    args.warmup_runs,
                    args.measured_runs,
                    device,
                )


if __name__ == "__main__":
    main()
