import csv
import gc
import importlib.util
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import triton

if __package__:
    from .kernel import flash_attention_forward
else:
    from kernel import flash_attention_forward


BATCH_SIZE = 2
NUM_HEADS = 8
HEAD_DIM = 64
SEQUENCE_LENGTHS = (128, 256, 512, 1024, 2048)
DTYPE = torch.float16
WARMUP_MS = 100
REP_MS = 300
QUANTILES = (0.2, 0.5, 0.8)

RESULTS_DIR = Path(__file__).resolve().parent / "benchmark_results"
CSV_PATH = RESULTS_DIR / "flash-attention-forward.csv"
LATENCY_PLOT_PATH = RESULTS_DIR / "flash-attention-forward-latency.png"
MEMORY_PLOT_PATH = RESULTS_DIR / "flash-attention-forward-memory.png"


@lru_cache(maxsize=1)
def _load_naive_attention_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "05_attention_naive"
        / "kernel.py"
    )
    spec = importlib.util.spec_from_file_location(
        "attention_naive_kernel_for_flash_benchmark",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 05 实现：{module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bench(provider) -> tuple[float, float, float]:
    values = triton.testing.do_bench(
        provider,
        warmup=WARMUP_MS,
        rep=REP_MS,
        quantiles=QUANTILES,
    )
    return tuple(float(value) for value in values)


def _measure_peak_allocated_mib(provider, device: torch.device) -> float:
    """测量 provider 相对已存在 Q/K/V 的活跃显存峰值增量。"""
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_bytes = torch.cuda.memory_allocated(device)

    result = provider()
    torch.cuda.synchronize(device)
    peak_bytes = torch.cuda.max_memory_allocated(device)

    del result
    return (peak_bytes - baseline_bytes) / (1024**2)


def _matmul_equivalent_tflops(
    sequence_length: int,
    latency_ms: float,
) -> float:
    # QK^T 和 P@V 两个 MatMul，共约 4*B*H*N^2*D FLOPs。
    flops = (
        4
        * BATCH_SIZE
        * NUM_HEADS
        * sequence_length
        * sequence_length
        * HEAD_DIM
    )
    return flops / latency_ms * 1e-9


def _tensor_sizes_mib(sequence_length: int) -> tuple[float, float]:
    dtype_bytes = torch.empty((), dtype=DTYPE).element_size()
    one_matrix_bytes = (
        BATCH_SIZE
        * NUM_HEADS
        * sequence_length
        * sequence_length
        * dtype_bytes
    )
    output_bytes = (
        BATCH_SIZE
        * NUM_HEADS
        * sequence_length
        * HEAD_DIM
        * dtype_bytes
    )
    return 2 * one_matrix_bytes / (1024**2), output_bytes / (1024**2)


def _write_csv(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_latency(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    providers = (
        ("torch", "#4472C4", "PyTorch Naive"),
        ("naive", "#70AD47", "Triton Naive"),
        ("flash", "#ED7D31", "Triton Flash"),
    )
    for axis, causal in zip(axes, (False, True)):
        subset = sorted(
            (row for row in rows if row["causal"] == causal),
            key=lambda row: row["sequence_length"],
        )
        sequence_lengths = [row["sequence_length"] for row in subset]
        for prefix, color, label in providers:
            p20 = [row[f"{prefix}_p20_ms"] for row in subset]
            p50 = [row[f"{prefix}_p50_ms"] for row in subset]
            p80 = [row[f"{prefix}_p80_ms"] for row in subset]
            axis.plot(sequence_lengths, p50, marker="o", color=color, label=label)
            axis.fill_between(sequence_lengths, p20, p80, color=color, alpha=0.16)

        axis.set_title("Causal" if causal else "Non-causal")
        axis.set_xlabel("Sequence length N")
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel("Latency (ms), P50 with P20-P80 band")
    axes[0].legend()
    figure.suptitle("Attention Forward Latency")
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
        for key, label, style in (
            ("naive_s_p_mib", "Theoretical Naive S+P", "-"),
            ("torch_peak_mib", "PyTorch Naive peak", "--"),
            ("naive_peak_mib", "Triton Naive peak", "--"),
            ("flash_peak_mib", "Triton Flash peak", "-."),
        ):
            axis.plot(
                sequence_lengths,
                [row[key] for row in subset],
                marker="o",
                linestyle=style,
                label=label,
            )

        axis.set_title("Causal" if causal else "Non-causal")
        axis.set_xlabel("Sequence length N")
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel("Peak allocated delta / capacity (MiB)")
    axes[0].legend()
    figure.suptitle("Attention Forward Intermediate Memory")
    figure.tight_layout()
    figure.savefig(MEMORY_PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_benchmark() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark 需要 CUDA GPU")

    device = torch.device("cuda")
    naive_module = _load_naive_attention_module()
    torch.manual_seed(0)
    rows = []

    print(
        f"B={BATCH_SIZE}, H={NUM_HEADS}, D={HEAD_DIM}, dtype={DTYPE}, "
        f"warmup={WARMUP_MS}ms, rep={REP_MS}ms"
    )
    print(
        "causal N    Torch P50  Naive P50  Flash P50  "
        "Torch/Flash  Naive/Flash  Flash peak(MiB)"
    )

    for causal in (False, True):
        for sequence_length in SEQUENCE_LENGTHS:
            shape = (BATCH_SIZE, NUM_HEADS, sequence_length, HEAD_DIM)
            q = torch.randn(shape, device=device, dtype=DTYPE)
            k = torch.randn(shape, device=device, dtype=DTYPE)
            v = torch.randn(shape, device=device, dtype=DTYPE)

            def torch_provider():
                return naive_module.naive_attention_reference(
                    q,
                    k,
                    v,
                    causal=causal,
                )[0]

            def naive_provider():
                return naive_module.naive_attention(q, k, v, causal=causal)

            def flash_provider():
                return flash_attention_forward(q, k, v, causal=causal)

            # 计时前完成 correctness、JIT 和缓存建立。
            expected = torch_provider()
            naive_output = naive_provider()
            flash_output = flash_provider()
            torch.testing.assert_close(naive_output, expected, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(flash_output, expected, atol=2e-2, rtol=2e-2)
            torch.cuda.synchronize(device)
            del expected, naive_output, flash_output

            peaks = {
                "torch": _measure_peak_allocated_mib(torch_provider, device),
                "naive": _measure_peak_allocated_mib(naive_provider, device),
                "flash": _measure_peak_allocated_mib(flash_provider, device),
            }
            quantiles = {
                "torch": _bench(torch_provider),
                "naive": _bench(naive_provider),
                "flash": _bench(flash_provider),
            }
            naive_s_p_mib, output_mib = _tensor_sizes_mib(sequence_length)

            row = {
                "causal": causal,
                "batch_size": BATCH_SIZE,
                "num_heads": NUM_HEADS,
                "sequence_length": sequence_length,
                "head_dim": HEAD_DIM,
                "dtype": str(DTYPE).removeprefix("torch."),
                "naive_s_p_mib": naive_s_p_mib,
                "output_mib": output_mib,
                "flash_theoretical_n2_intermediate_mib": 0.0,
            }
            for prefix in ("torch", "naive", "flash"):
                p20, p50, p80 = quantiles[prefix]
                row[f"{prefix}_p20_ms"] = p20
                row[f"{prefix}_p50_ms"] = p50
                row[f"{prefix}_p80_ms"] = p80
                row[f"{prefix}_tflops"] = _matmul_equivalent_tflops(
                    sequence_length,
                    p50,
                )
                row[f"{prefix}_peak_mib"] = peaks[prefix]

            row["torch_over_flash_speedup"] = (
                row["torch_p50_ms"] / row["flash_p50_ms"]
            )
            row["naive_over_flash_speedup"] = (
                row["naive_p50_ms"] / row["flash_p50_ms"]
            )
            rows.append(row)

            print(
                f"{str(causal):<7} {sequence_length:<4d} "
                f"{row['torch_p50_ms']:>10.4f} "
                f"{row['naive_p50_ms']:>10.4f} "
                f"{row['flash_p50_ms']:>10.4f} "
                f"{row['torch_over_flash_speedup']:>11.3f}x "
                f"{row['naive_over_flash_speedup']:>11.3f}x "
                f"{row['flash_peak_mib']:>15.3f}"
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
