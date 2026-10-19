"""扫描 GPT-2 Prefill/Decode 的延迟、吞吐和峰值显存。

本脚本使用预分配 KV Cache，重点观察 batch、prompt length 和 output length
如何影响一条可复现的单卡推理路径。Profiler 是可选项，单独运行且不纳入计时。
"""

import argparse
import itertools
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import GPT2LMHeadModel

# 脚本放在 benchmarks/ 子目录，但模型和 Decode 实现在上一级目录。
# 把项目根目录加入模块搜索路径后，从项目根目录直接运行脚本也能找到它们。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decode import AttentionImpl, KVCacheList, _decode_block, _embed, _prefill_block
from model import GPT2ReferenceModel


def _sync(device: torch.device) -> None:
    """CUDA 调用通常是异步提交的，计时前后要等 GPU 真正做完。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_ms(start: float, device: torch.device) -> float:
    _sync(device)
    return (time.perf_counter() - start) * 1000.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    """用 nearest-rank 计算百分位，样本较少时行为也容易解释。"""
    ordered = sorted(values)
    index = max(0, int((percentile / 100.0) * len(ordered) + 0.999999) - 1)
    return ordered[min(index, len(ordered) - 1)]


def _make_input_ids(batch_size: int, prompt_len: int, device: torch.device) -> torch.Tensor:
    """造一批合法、可复现的 token ID，不把 tokenizer 时间算进模型计时。"""
    token_ids = (torch.arange(prompt_len, device=device) + 100).remainder(50257)
    return token_ids.unsqueeze(0).expand(batch_size, -1).contiguous()


@torch.no_grad()
def _run_prealloc(
    model: GPT2ReferenceModel,
    input_ids: torch.Tensor,
    output_tokens: int,
    prefill_attention_impl: Optional[AttentionImpl] = None,
) -> Tuple[float, float, List[float]]:
    """返回 Prefill、首 token 采样和后续 Decode 的耗时。"""
    device = input_ids.device
    batch_size, prompt_len = input_ids.shape
    max_seq_len = prompt_len + output_tokens

    # Prefill 包含完整 prompt 的 embedding、各层计算、KV Cache 建立，
    # 以及完整 logits 的输出。与前一节基线保持相同计算边界。
    _sync(device)
    start = time.perf_counter()
    with torch.profiler.record_function("prefill"):
        positions = torch.arange(prompt_len, device=device).unsqueeze(0)
        hidden_states = _embed(model, input_ids, positions)
        caches: KVCacheList = []

        for block in model.transformer.h:
            hidden_states, cache = _prefill_block(
                block,
                hidden_states,
                mode="prealloc",
                max_seq_len=max_seq_len,
                attention_impl=prefill_attention_impl,
            )
            caches.append(cache)

        hidden_states = model.transformer.ln_f(hidden_states)
        logits = model.lm_head(hidden_states)
        next_logits = logits[:, -1, :]
    prefill_ms = _elapsed_ms(start, device)

    # Prefill 的最后位置负责预测第一个新 token。把采样时间单列出来，
    # 这样 Prefill latency 和完整首 token 延迟（TTFT）不会混为一谈。
    _sync(device)
    start = time.perf_counter()
    next_token = next_logits.argmax(dim=-1, keepdim=True)
    first_sample_ms = _elapsed_ms(start, device)

    # 后续 output_tokens - 1 个 token 各测一次增量 Decode。
    decode_ms: List[float] = []
    for step in range(output_tokens - 1):
        position = prompt_len + step
        token_position = torch.full(
            (batch_size, 1),
            position,
            dtype=torch.long,
            device=device,
        )

        _sync(device)
        start = time.perf_counter()
        with torch.profiler.record_function("decode_step"):
            hidden_states = _embed(model, next_token, token_position)
            updated_caches: KVCacheList = []

            for block, cache in zip(model.transformer.h, caches):
                hidden_states, cache = _decode_block(
                    block,
                    hidden_states,
                    cache,
                    position,
                    mode="prealloc",
                )
                updated_caches.append(cache)

            caches = updated_caches
            hidden_states = model.transformer.ln_f(hidden_states)
            next_logits = model.lm_head(hidden_states[:, -1, :])
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        decode_ms.append(_elapsed_ms(start, device))

    return prefill_ms, first_sample_ms, decode_ms


def _load_model(dtype: torch.dtype, device: torch.device) -> GPT2ReferenceModel:
    """从 Hugging Face GPT-2 权重创建手写模型；参考模型留在 CPU。"""
    reference_model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=dtype)
    reference_model.eval()

    model = GPT2ReferenceModel(reference_model.config)
    missing_keys, unexpected_keys = model.load_state_dict(
        reference_model.state_dict(),
        strict=False,
    )
    if missing_keys:
        raise RuntimeError(f"missing model weights: {missing_keys}")
    if any(not key.endswith("masked_bias") for key in unexpected_keys):
        raise RuntimeError(f"unexpected model weights: {unexpected_keys}")

    del reference_model
    return model.to(device=device, dtype=dtype).eval()


def _measure_case(
    model: GPT2ReferenceModel,
    batch_size: int,
    prompt_len: int,
    output_tokens: int,
    warmup_runs: int,
    measured_runs: int,
    device: torch.device,
) -> Dict[str, float]:
    """先 warmup，再多次测同一 shape，并记录 allocated 峰值。"""
    input_ids = _make_input_ids(batch_size, prompt_len, device)

    for _ in range(warmup_runs):
        _run_prealloc(model, input_ids, output_tokens)

    prefill_samples: List[float] = []
    first_sample_samples: List[float] = []
    decode_samples: List[float] = []
    peak_extra_samples: List[float] = []
    peak_reserved_samples: List[float] = []

    for _ in range(measured_runs):
        # baseline 是模型和输入已经驻留后的活跃显存；额外峰值包含本次
        # Cache、logits 和中间张量，不把模型本身重复算进工作区增量。
        _sync(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        prefill_ms, first_sample_ms, decode_ms = _run_prealloc(
            model,
            input_ids,
            output_tokens,
        )
        prefill_samples.append(prefill_ms)
        first_sample_samples.append(first_sample_ms)
        decode_samples.extend(decode_ms)
        peak_extra_samples.append(
            (torch.cuda.max_memory_allocated(device) - baseline_allocated) / (1024 ** 2)
        )
        peak_reserved_samples.append(torch.cuda.max_memory_reserved(device) / (1024 ** 2))

    prefill_p50 = statistics.median(prefill_samples)
    first_sample_p50 = statistics.median(first_sample_samples)
    decode_p50 = statistics.median(decode_samples) if decode_samples else 0.0
    decode_p95 = _percentile(decode_samples, 95) if decode_samples else 0.0
    if output_tokens > 1:
        decode_totals = [
            sum(decode_samples[index : index + output_tokens - 1])
            for index in range(0, len(decode_samples), output_tokens - 1)
        ]
        decode_total_p50 = statistics.median(decode_totals)
    else:
        decode_total_p50 = 0.0

    # TTFT 是 Prefill 结束后选出首 token 的时间；端到端吞吐把完整输出
    # token 数除以从 Prefill 开始到最后一个输出 token 的总时间。
    ttft_ms = prefill_p50 + first_sample_p50
    end_to_end_ms = ttft_ms + decode_total_p50
    # batch 中每条序列都会生成 output_tokens 个 token；吞吐统计总输出量。
    tokens_per_second = batch_size * output_tokens * 1000.0 / end_to_end_ms

    return {
        "prefill_p50_ms": prefill_p50,
        "ttft_p50_ms": ttft_ms,
        "decode_step_p50_ms": decode_p50,
        "decode_step_p95_ms": decode_p95,
        "tpot_p50_ms": decode_total_p50 / (output_tokens - 1) if output_tokens > 1 else 0.0,
        "tokens_per_second": tokens_per_second,
        "peak_extra_mib": statistics.median(peak_extra_samples),
        "peak_reserved_mib": max(peak_reserved_samples),
    }


def _profile_case(
    model: GPT2ReferenceModel,
    batch_size: int,
    prompt_len: int,
    output_tokens: int,
    device: torch.device,
    trace_dir: Path,
) -> None:
    """单独采一次 profiler trace；Profiler 下的耗时不用于性能表。"""
    trace_dir.mkdir(parents=True, exist_ok=True)
    input_ids = _make_input_ids(batch_size, prompt_len, device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        _run_prealloc(model, input_ids, output_tokens)

    trace_path = trace_dir / f"gpt2_b{batch_size}_p{prompt_len}_o{output_tokens}.json"
    profiler.export_chrome_trace(str(trace_path))
    operator_stats = profiler.key_averages()
    has_cuda_timing = any(
        getattr(operator, "self_cuda_time_total", 0.0) > 0
        for operator in operator_stats
    )

    if device.type == "cuda" and not has_cuda_timing:
        print("\nWARNING: no CUDA kernel timings were collected; showing CPU-side operator costs only.")
        sort_key = "self_cpu_time_total"
    else:
        sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"

    print("Profiler top operators (diagnostic only; do not compare these timings to benchmark):")
    print(operator_stats.table(sort_by=sort_key, row_limit=20))
    print(f"Chrome trace: {trace_path}")


def _parse_int_list(value: str) -> List[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated positive integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=_parse_int_list, default=[1, 2])
    parser.add_argument("--prompt-lengths", type=_parse_int_list, default=[32, 128])
    parser.add_argument("--output-lengths", type=_parse_int_list, default=[8, 32])
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--measured-runs", type=int, default=10)
    parser.add_argument("--profile", action="store_true", help="额外采集首组 shape 的 profiler trace")
    parser.add_argument("--trace-dir", type=Path, default=Path("profiles"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要 CUDA，请在已配置 GPU 的 WSL 环境运行。")
    if args.warmup_runs < 0 or args.measured_runs <= 0:
        raise ValueError("warmup-runs 不能小于 0，measured-runs 必须大于 0。")

    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    model = _load_model(dtype, device)
    cases = list(itertools.product(args.batch_sizes, args.prompt_lengths, args.output_lengths))

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        f"dtype: {args.dtype}; cache: preallocated; "
        f"warmup: {args.warmup_runs}; measured: {args.measured_runs}"
    )
    print("Prefill includes full prompt logits; TTFT adds first-token argmax; Decode step includes next-token argmax.")
    print("tokens/s counts output tokens from every sequence in the batch.")
    print("Peak extra is max_memory_allocated minus model/input baseline; reserved is reported separately.")
    print("\nB | prompt | output | Prefill p50 | TTFT p50 | TPOT p50 | step p95 | tokens/s | peak extra | peak reserved")

    for batch_size, prompt_len, output_tokens in cases:
        result = _measure_case(
            model,
            batch_size,
            prompt_len,
            output_tokens,
            args.warmup_runs,
            args.measured_runs,
            device,
        )
        print(
            f"{batch_size:>1} | {prompt_len:>6} | {output_tokens:>6} | "
            f"{result['prefill_p50_ms']:>11.3f} ms | {result['ttft_p50_ms']:>8.3f} ms | "
            f"{result['tpot_p50_ms']:>8.3f} ms | {result['decode_step_p95_ms']:>8.3f} ms | "
            f"{result['tokens_per_second']:>8.2f} | {result['peak_extra_mib']:>9.2f} MiB | "
            f"{result['peak_reserved_mib']:>12.2f} MiB"
        )

    if args.profile:
        _profile_case(model, *cases[0], device, args.trace_dir)


if __name__ == "__main__":
    main()
