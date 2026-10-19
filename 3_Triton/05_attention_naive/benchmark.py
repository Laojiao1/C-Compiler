import csv
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import triton

if __package__:
    from .kernel import naive_attention, naive_attention_reference
else:
    # 兼容直接执行：python benchmark.py
    from kernel import naive_attention, naive_attention_reference


BATCH_SIZE = 2
NUM_HEADS = 8
HEAD_DIM = 64
SEQUENCE_LENGTHS = (128, 256, 512, 1024, 2048)
DTYPE = torch.float16
WARMUP_MS = 100
REP_MS = 300
QUANTILES = (0.2, 0.5, 0.8)
RESULTS_DIR = Path(__file__).resolve().parent / "benchmark_results"
CSV_PATH = RESULTS_DIR / "attention-naive.csv"
LATENCY_PLOT_PATH = RESULTS_DIR / "attention-naive-latency.png"
MEMORY_PLOT_PATH = RESULTS_DIR / "attention-naive-memory.png"


def _intermediate_sizes_mib(
    batch_size: int,
    num_heads: int,
    sequence_length: int,
    dtype: torch.dtype,
) -> tuple[float, float, float]:
    """返回单个 S、单个 P 和 S+P 的理论容量。"""
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    matrix_bytes = (
        batch_size
        * num_heads
        * sequence_length
        * sequence_length
        * dtype_bytes
    )
    score_mib = matrix_bytes / (1024**2)
    probability_mib = score_mib
    return score_mib, probability_mib, score_mib + probability_mib


def _measure_peak_allocated_mib(provider, device: torch.device) -> float:
    """测量一次稳态调用相对当前活跃张量的峰值显存增量。"""
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_bytes = torch.cuda.memory_allocated(device)

    result = provider()
    torch.cuda.synchronize(device)
    peak_bytes = torch.cuda.max_memory_allocated(device)

    del result
    return (peak_bytes - baseline_bytes) / (1024**2)


def _bench_provider(provider) -> tuple[float, float, float]:
    measurements = triton.testing.do_bench(
        provider,
        warmup=WARMUP_MS,
        rep=REP_MS,
        quantiles=QUANTILES,
    )
    return tuple(float(value) for value in measurements)


def _format_quantiles(values: tuple[float, float, float]) -> str:
    return "/".join(f"{value:.4f}" for value in values)


def _write_csv(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_latency(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for axis, causal in zip(axes, (False, True)):
        subset = sorted(
            (row for row in rows if row["causal"] == causal),
            key=lambda row: row["sequence_length"],
        )
        sequence_lengths = [row["sequence_length"] for row in subset]

        for provider, color, label in (
            ("torch", "#4472C4", "PyTorch Naive"),
            ("triton", "#ED7D31", "Triton Naive"),
        ):
            p20 = [row[f"{provider}_p20_ms"] for row in subset]
            p50 = [row[f"{provider}_p50_ms"] for row in subset]
            p80 = [row[f"{provider}_p80_ms"] for row in subset]
            axis.plot(
                sequence_lengths,
                p50,
                marker="o",
                linewidth=2,
                color=color,
                label=label,
            )
            axis.fill_between(sequence_lengths, p20, p80, color=color, alpha=0.18)

        axis.set_title("Causal" if causal else "Non-causal")
        axis.set_xlabel("Sequence length N")
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel("Latency (ms), P50 with P20-P80 band")
    axes[0].legend()
    figure.suptitle("Naive Attention Latency vs Sequence Length")
    figure.tight_layout()
    figure.savefig(LATENCY_PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_memory(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for axis, causal in zip(axes, (False, True)):
        subset = sorted(
            (row for row in rows if row["causal"] == causal),
            key=lambda row: row["sequence_length"],
        )
        sequence_lengths = [row["sequence_length"] for row in subset]
        axis.plot(
            sequence_lengths,
            [row["score_mib"] for row in subset],
            marker="o",
            linewidth=2,
            label="One S matrix",
        )
        axis.plot(
            sequence_lengths,
            [row["score_probability_mib"] for row in subset],
            marker="o",
            linewidth=2,
            label="Theoretical S + P",
        )
        axis.plot(
            sequence_lengths,
            [row["torch_peak_mib"] for row in subset],
            marker="s",
            linestyle="--",
            label="PyTorch peak delta",
        )
        axis.plot(
            sequence_lengths,
            [row["triton_peak_mib"] for row in subset],
            marker="s",
            linestyle="--",
            label="Triton peak delta",
        )
        axis.set_title("Causal" if causal else "Non-causal")
        axis.set_xlabel("Sequence length N")
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel("Memory (MiB)")
    axes[0].legend()
    figure.suptitle("Naive Attention Explicit Intermediate Memory")
    figure.tight_layout()
    figure.savefig(MEMORY_PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_benchmark() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark 需要 CUDA GPU")

    device = torch.device("cuda")
    torch.manual_seed(0)

    print(
        "固定参数："
        f"B={BATCH_SIZE}, H={NUM_HEADS}, D={HEAD_DIM}, "
        f"dtype={DTYPE}, warmup={WARMUP_MS} ms, rep={REP_MS} ms"
    )
    print(
        "causal  N     Torch P20/P50/P80(ms)  "
        "Triton P20/P50/P80(ms)  speedup  S(MiB)  S+P(MiB)  "
        "Torch peak(MiB)  Triton peak(MiB)"
    )
    rows = []

    for causal in (False, True):
        for sequence_length in SEQUENCE_LENGTHS:
            shape = (
                BATCH_SIZE,
                NUM_HEADS,
                sequence_length,
                HEAD_DIM,
            )
            q = torch.randn(shape, device=device, dtype=DTYPE)
            k = torch.randn(shape, device=device, dtype=DTYPE)
            v = torch.randn(shape, device=device, dtype=DTYPE)

            def torch_provider():
                return naive_attention_reference(
                    q,
                    k,
                    v,
                    causal=causal,
                )[0]

            def triton_provider():
                return naive_attention(q, k, v, causal=causal)

            # 正式计时前先验证结果，同时完成 Triton JIT 编译和缓存。
            expected = torch_provider()
            actual = triton_provider()
            torch.testing.assert_close(
                actual,
                expected,
                atol=2e-2,
                rtol=2e-2,
            )
            torch.cuda.synchronize(device)
            del expected, actual

            torch_peak_mib = _measure_peak_allocated_mib(torch_provider, device)
            triton_peak_mib = _measure_peak_allocated_mib(triton_provider, device)
            torch_quantiles = _bench_provider(torch_provider)
            triton_quantiles = _bench_provider(triton_provider)

            torch_p50 = torch_quantiles[1]
            triton_p50 = triton_quantiles[1]
            speedup = torch_p50 / triton_p50
            score_mib, probability_mib, total_intermediate_mib = (
                _intermediate_sizes_mib(
                BATCH_SIZE,
                NUM_HEADS,
                sequence_length,
                DTYPE,
                )
            )

            rows.append(
                {
                    "causal": causal,
                    "batch_size": BATCH_SIZE,
                    "num_heads": NUM_HEADS,
                    "sequence_length": sequence_length,
                    "head_dim": HEAD_DIM,
                    "dtype": str(DTYPE).removeprefix("torch."),
                    "torch_p20_ms": torch_quantiles[0],
                    "torch_p50_ms": torch_quantiles[1],
                    "torch_p80_ms": torch_quantiles[2],
                    "triton_p20_ms": triton_quantiles[0],
                    "triton_p50_ms": triton_quantiles[1],
                    "triton_p80_ms": triton_quantiles[2],
                    "speedup": speedup,
                    "score_mib": score_mib,
                    "probability_mib": probability_mib,
                    "score_probability_mib": total_intermediate_mib,
                    "torch_peak_mib": torch_peak_mib,
                    "triton_peak_mib": triton_peak_mib,
                }
            )

            print(
                f"{str(causal):<7} "
                f"{sequence_length:<5d} "
                f"{_format_quantiles(torch_quantiles):<24} "
                f"{_format_quantiles(triton_quantiles):<25} "
                f"{speedup:>7.3f}x "
                f"{score_mib:>7.1f} "
                f"{total_intermediate_mib:>10.1f} "
                f"{torch_peak_mib:>15.1f} "
                f"{triton_peak_mib:>16.1f}"
            )

            del q, k, v

    _write_csv(rows)
    _plot_latency(rows)
    _plot_memory(rows)
    print(f"CSV: {CSV_PATH}")
    print(f"Latency plot: {LATENCY_PLOT_PATH}")
    print(f"Memory plot: {MEMORY_PLOT_PATH}")


if __name__ == "__main__":
    run_benchmark()
