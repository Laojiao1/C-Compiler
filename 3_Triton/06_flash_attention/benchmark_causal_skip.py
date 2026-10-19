import csv
import importlib.util
import random
import statistics
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import triton

if __package__:
    from .kernel import flash_attention_forward_configured
else:
    from kernel import flash_attention_forward_configured


BATCH_SIZE = 2
NUM_HEADS = 8
HEAD_DIM = 64
SEQUENCE_LENGTHS = (128, 256, 512, 1024, 2048)
DTYPE = torch.float16
BLOCK_M = 64
BLOCK_N = 32
NUM_WARPS = 4
ROUNDS = 3
WARMUP_MS = 50
REP_MS = 150
QUANTILES = (0.2, 0.5, 0.8)
PROVIDERS = ("all_tiles", "skip_future")

RESULTS_DIR = Path(__file__).resolve().parent / "benchmark_results"
RAW_CSV_PATH = RESULTS_DIR / "flash-attention-causal-skip-raw.csv"
SUMMARY_CSV_PATH = RESULTS_DIR / "flash-attention-causal-skip-summary.csv"
PLOT_PATH = RESULTS_DIR / "flash-attention-causal-skip.png"


@lru_cache(maxsize=1)
def _load_naive_reference():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "05_attention_naive"
        / "kernel.py"
    )
    spec = importlib.util.spec_from_file_location(
        "attention_naive_kernel_for_causal_skip",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 05 reference：{module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.naive_attention_reference


def _make_provider(q, k, v, skip_future: bool):
    def provider():
        return flash_attention_forward_configured(
            q,
            k,
            v,
            causal=True,
            block_m=BLOCK_M,
            block_n=BLOCK_N,
            num_warps=NUM_WARPS,
            skip_causal_future_tiles=skip_future,
        )

    return provider


def _bench(provider) -> tuple[float, float, float]:
    values = triton.testing.do_bench(
        provider,
        warmup=WARMUP_MS,
        rep=REP_MS,
        quantiles=QUANTILES,
    )
    return tuple(float(value) for value in values)


def _tile_pair_count(sequence_length: int, skip_future: bool) -> int:
    query_blocks = triton.cdiv(sequence_length, BLOCK_M)
    key_blocks = triton.cdiv(sequence_length, BLOCK_N)
    if not skip_future:
        return query_blocks * key_blocks

    return sum(
        triton.cdiv(
            min((query_block_idx + 1) * BLOCK_M, sequence_length),
            BLOCK_N,
        )
        for query_block_idx in range(query_blocks)
    )


def _full_equivalent_flops(sequence_length: int) -> int:
    return (
        4
        * BATCH_SIZE
        * NUM_HEADS
        * sequence_length
        * sequence_length
        * HEAD_DIM
    )


def _executed_tile_flops(sequence_length: int, skip_future: bool) -> int:
    return (
        4
        * BATCH_SIZE
        * NUM_HEADS
        * _tile_pair_count(sequence_length, skip_future)
        * BLOCK_M
        * BLOCK_N
        * HEAD_DIM
    )


def _tflops(flops: int, latency_ms: float) -> float:
    return flops / latency_ms * 1e-9


def _write_csv(path: Path, rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(raw_rows: list[dict]) -> list[dict]:
    summary_rows = []
    for sequence_length in SEQUENCE_LENGTHS:
        for provider in PROVIDERS:
            group = [
                row
                for row in raw_rows
                if row["sequence_length"] == sequence_length
                and row["provider"] == provider
            ]
            p50_values = [float(row["p50_ms"]) for row in group]
            speedups = [float(row["skip_speedup"]) for row in group]
            full_tflops = [float(row["full_equivalent_tflops"]) for row in group]
            executed_tflops = [float(row["executed_tile_tflops"]) for row in group]
            summary_rows.append(
                {
                    "sequence_length": sequence_length,
                    "provider": provider,
                    "rounds": len(group),
                    "executed_tile_fraction": group[0]["executed_tile_fraction"],
                    "median_p50_ms": statistics.median(p50_values),
                    "min_p50_ms": min(p50_values),
                    "max_p50_ms": max(p50_values),
                    "median_skip_speedup": statistics.median(speedups),
                    "min_skip_speedup": min(speedups),
                    "max_skip_speedup": max(speedups),
                    "median_full_equivalent_tflops": statistics.median(full_tflops),
                    "median_executed_tile_tflops": statistics.median(executed_tflops),
                }
            )
    return summary_rows


def _plot(summary_rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for provider, label in (
        ("all_tiles", "All causal tiles"),
        ("skip_future", "Skip fully-future tiles"),
    ):
        subset = [row for row in summary_rows if row["provider"] == provider]
        axes[0].plot(
            SEQUENCE_LENGTHS,
            [row["median_p50_ms"] for row in subset],
            marker="o",
            label=label,
        )

    skip_rows = [
        row for row in summary_rows if row["provider"] == "skip_future"
    ]
    axes[1].plot(
        SEQUENCE_LENGTHS,
        [row["median_skip_speedup"] for row in skip_rows],
        marker="o",
        label="Measured speedup",
    )
    axes[1].plot(
        SEQUENCE_LENGTHS,
        [1.0 / float(row["executed_tile_fraction"]) for row in skip_rows],
        marker="o",
        linestyle="--",
        label="Ideal inverse work fraction",
    )

    axes[0].set_title("Causal latency")
    axes[0].set_xlabel("Sequence length N")
    axes[0].set_ylabel("Median P50 latency (ms)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].axhline(1.0, color="black", linewidth=1, linestyle=":")
    axes[1].set_title("Skipping benefit vs reduced tile work")
    axes[1].set_xlabel("Sequence length N")
    axes[1].set_ylabel("Speedup")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.suptitle("FlashAttention Causal Fully-Future Tile Skipping")
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_causal_skip_benchmark() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark 需要 CUDA GPU")

    device = torch.device("cuda")
    reference = _load_naive_reference()
    raw_rows = []
    torch.manual_seed(0)

    print(
        f"B={BATCH_SIZE}, H={NUM_HEADS}, D={HEAD_DIM}, dtype={DTYPE}, "
        f"tile={BLOCK_M}x{BLOCK_N}, w{NUM_WARPS}, rounds={ROUNDS}"
    )

    for sequence_length in SEQUENCE_LENGTHS:
        shape = (BATCH_SIZE, NUM_HEADS, sequence_length, HEAD_DIM)
        q = torch.randn(shape, device=device, dtype=DTYPE)
        k = torch.randn(shape, device=device, dtype=DTYPE)
        v = torch.randn(shape, device=device, dtype=DTYPE)
        expected = reference(q, k, v, causal=True)[0]
        providers = {
            "all_tiles": _make_provider(q, k, v, False),
            "skip_future": _make_provider(q, k, v, True),
        }

        for provider in providers.values():
            actual = provider()
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
            del actual
        torch.cuda.synchronize(device)

        for round_idx in range(ROUNDS):
            order = list(PROVIDERS)
            random.Random(2026 + sequence_length + round_idx).shuffle(order)
            round_rows = []

            for measurement_order, provider_name in enumerate(order):
                skip_future = provider_name == "skip_future"
                p20, p50, p80 = _bench(providers[provider_name])
                full_flops = _full_equivalent_flops(sequence_length)
                executed_flops = _executed_tile_flops(
                    sequence_length,
                    skip_future,
                )
                all_tile_pairs = _tile_pair_count(sequence_length, False)
                executed_tile_pairs = _tile_pair_count(
                    sequence_length,
                    skip_future,
                )
                round_rows.append(
                    {
                        "sequence_length": sequence_length,
                        "round": round_idx,
                        "measurement_order": measurement_order,
                        "provider": provider_name,
                        "executed_tile_pairs_per_bh": executed_tile_pairs,
                        "executed_tile_fraction": executed_tile_pairs / all_tile_pairs,
                        "p20_ms": p20,
                        "p50_ms": p50,
                        "p80_ms": p80,
                        "full_equivalent_tflops": _tflops(full_flops, p50),
                        "executed_tile_tflops": _tflops(executed_flops, p50),
                        "skip_speedup": 1.0,
                    }
                )

            baseline_p50 = next(
                float(row["p50_ms"])
                for row in round_rows
                if row["provider"] == "all_tiles"
            )
            for row in round_rows:
                row["skip_speedup"] = baseline_p50 / float(row["p50_ms"])
                print(
                    f"N={sequence_length:<4d} round={round_idx} "
                    f"order={row['measurement_order']} {row['provider']:<11} "
                    f"tiles={row['executed_tile_fraction']:.3f} "
                    f"P50={float(row['p50_ms']):.4f}ms "
                    f"speedup={float(row['skip_speedup']):.3f}x"
                )
            raw_rows.extend(round_rows)

        del expected, q, k, v

    summary_rows = _summarize(raw_rows)
    _write_csv(RAW_CSV_PATH, raw_rows)
    _write_csv(SUMMARY_CSV_PATH, summary_rows)
    _plot(summary_rows)
    print(f"Raw CSV: {RAW_CSV_PATH}")
    print(f"Summary CSV: {SUMMARY_CSV_PATH}")
    print(f"Plot: {PLOT_PATH}")


if __name__ == "__main__":
    run_causal_skip_benchmark()
