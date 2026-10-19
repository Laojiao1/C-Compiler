"""测量 GPT-2 三条 Decode 路径的 Prefill 和逐 token 延迟。

这是 5.1.3 的单配置基线，不做 batch、prompt length 或 output length 扫描。
完整 workload sweep 和 profiler 分析留到 5.1.5。
"""

import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import GPT2LMHeadModel

# 脚本放在 benchmarks/ 子目录，但模型和 Decode 实现在上一级目录。
# 这样无论从项目根目录还是 benchmarks/ 目录启动，都能导入项目模块。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decode import CacheMode, KVCacheList, _decode_block, _embed, _prefill_block
from model import GPT2ReferenceModel


def _sync(device: torch.device) -> None:
    """CUDA 是异步执行的；计时边界需要等 GPU 工作真正完成。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_ms(start: float, device: torch.device) -> float:
    _sync(device)
    return (time.perf_counter() - start) * 1000.0


@torch.no_grad()
def _measure_without_cache(
    model: GPT2ReferenceModel,
    input_ids: torch.Tensor,
    output_tokens: int,
) -> Tuple[float, List[float]]:
    """路径 A：首次 forward 是 Prefill，其余 forward 每次重算完整序列。"""
    device = input_ids.device

    _sync(device)
    start = time.perf_counter()
    logits = model(input_ids)
    prefill_ms = _elapsed_ms(start, device)

    sequence = input_ids
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    decode_ms: List[float] = []

    # Prefill 的最后位置已经给出第一个新 token。
    # 后续每生成一个 token，都重新计算 prompt 和已生成的全部 token。
    for _ in range(output_tokens - 1):
        sequence = torch.cat((sequence, next_token), dim=1)

        _sync(device)
        start = time.perf_counter()
        logits = model(sequence)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decode_ms.append(_elapsed_ms(start, device))

    return prefill_ms, decode_ms


@torch.no_grad()
def _measure_with_cache(
    model: GPT2ReferenceModel,
    input_ids: torch.Tensor,
    output_tokens: int,
    mode: CacheMode,
) -> Tuple[float, List[float]]:
    """路径 B/C：Prefill 建立 Cache，后续每次只计算一个 token。"""
    device = input_ids.device
    batch_size, prompt_len = input_ids.shape
    max_seq_len = prompt_len + output_tokens

    _sync(device)
    start = time.perf_counter()

    # Prefill 对所有路径都处理完整 prompt；这里同时生成每层初始 K/V。
    prompt_positions = torch.arange(prompt_len, device=device).unsqueeze(0)
    hidden_states = _embed(model, input_ids, prompt_positions)
    caches: KVCacheList = []

    for block in model.transformer.h:
        hidden_states, cache = _prefill_block(
            block,
            hidden_states,
            mode,
            max_seq_len,
        )
        caches.append(cache)

    hidden_states = model.transformer.ln_f(hidden_states)
    # 与无缓存路径一样，对整段 prompt 计算 logits，再取最后一位。
    # 这样 Prefill 计时的工作量边界保持一致。
    logits = model.lm_head(hidden_states)
    logits = logits[:, -1, :]
    prefill_ms = _elapsed_ms(start, device)

    # Prefill logits 预测第一个新 token。之后的每次迭代处理刚生成的 token，
    # 得到下一个 token 的 logits，所以单步 Decode 计时共 output_tokens - 1 次。
    next_token = logits.argmax(dim=-1, keepdim=True)
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

        hidden_states = _embed(model, next_token, token_position)
        updated_caches: KVCacheList = []

        for block, cache in zip(model.transformer.h, caches):
            hidden_states, cache = _decode_block(
                block,
                hidden_states,
                cache,
                position,
                mode,
            )
            updated_caches.append(cache)

        caches = updated_caches
        hidden_states = model.transformer.ln_f(hidden_states)
        logits = model.lm_head(hidden_states[:, -1, :])
        next_token = logits.argmax(dim=-1, keepdim=True)
        decode_ms.append(_elapsed_ms(start, device))

    return prefill_ms, decode_ms


def _run_once(
    mode: str,
    model: GPT2ReferenceModel,
    input_ids: torch.Tensor,
    output_tokens: int,
) -> Tuple[float, List[float]]:
    if mode == "no_cache":
        return _measure_without_cache(model, input_ids, output_tokens)
    if mode == "cat":
        return _measure_with_cache(model, input_ids, output_tokens, "cat")
    if mode == "prealloc":
        return _measure_with_cache(model, input_ids, output_tokens, "prealloc")
    raise ValueError(f"unknown Decode path: {mode}")


def _percentile(values: List[float], percentile: float) -> float:
    """用排序后的 nearest-rank 方法计算百分位数。"""
    ordered = sorted(values)
    index = max(0, int((percentile / 100.0) * len(ordered) + 0.999999) - 1)
    return ordered[min(index, len(ordered) - 1)]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("这组基线需要 CUDA，请在可用的 WSL GPU 环境运行。")

    device = torch.device("cuda")
    reference_model = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        torch_dtype=torch.float32,
    )
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

    model = model.to(device).eval()
    input_ids = torch.tensor(
        [[15496, 11, 995, 318, 257]],
        dtype=torch.long,
        device=device,
    )

    # 这是 5.1.3 的小规模基线，不是最终性能扫描。
    output_tokens = 8
    warmup_runs = 10
    measured_runs = 50
    modes = ("no_cache", "cat", "prealloc")
    results: Dict[str, Tuple[List[float], List[float]]] = {}

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"dtype: float32, batch: 1, prompt tokens: {input_ids.shape[1]}")
    print(f"output tokens: {output_tokens}")
    print(f"warmup runs: {warmup_runs}, measured runs: {measured_runs}")
    print("Decode latency covers tokens 2..N; token 1 is produced by Prefill.")

    for mode in modes:
        # Warmup 单独执行，不纳入结果，避免首次 kernel 初始化影响测量。
        for _ in range(warmup_runs):
            _run_once(mode, model, input_ids, output_tokens)

        prefill_samples: List[float] = []
        decode_samples: List[float] = []

        for _ in range(measured_runs):
            prefill_ms, decode_ms = _run_once(
                mode,
                model,
                input_ids,
                output_tokens,
            )
            prefill_samples.append(prefill_ms)
            decode_samples.extend(decode_ms)

        results[mode] = (prefill_samples, decode_samples)

    print("\nResults (milliseconds; report these as a local baseline only):")
    for mode, (prefill_samples, decode_samples) in results.items():
        print(
            f"{mode:>8} | "
            f"Prefill p50={statistics.median(prefill_samples):.3f} | "
            f"Decode step p50={statistics.median(decode_samples):.3f} | "
            f"Decode step p95={_percentile(decode_samples, 95):.3f}"
        )


if __name__ == "__main__":
    main()
