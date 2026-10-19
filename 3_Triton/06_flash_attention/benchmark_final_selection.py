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
SEQUENCE_LENGTHS = (256, 1024, 2048)
DTYPE = torch.float16
ROUNDS = 3
WARMUP_MS = 50
REP_MS = 150
QUANTILES = (0.2, 0.5, 0.8)

BASELINE = (32, 32, 4)
CANDIDATES = (
    BASELINE,
    (32, 64, 2),
    (32, 64, 4),
    (64, 32, 2),
    (64, 32, 4),
    (64, 32, 8),
    (64, 64, 2),
    (64, 64, 4),
    (64, 64, 8),
)

RESULTS_DIR = Path(__file__).resolve().parent / "benchmark_results"
RAW_CSV_PATH = RESULTS_DIR / "flash-attention-final-selection-raw.csv"
SUMMARY_CSV_PATH = RESULTS_DIR / "flash-attention-final-selection-summary.csv"
PLOT_PATH = RESULTS_DIR / "flash-attention-final-selection.png"


@lru_cache(maxsize=1)
def _load_naive_reference():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "05_attention_naive"
        / "kernel.py"
    )
    spec = importlib.util.spec_from_file_location(
        "attention_naive_kernel_for_final_selection",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 05 reference：{module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.naive_attention_reference


def _config_name(config: tuple[int, int, int]) -> str:
    block_m, block_n, num_warps = config
    return f"m{block_m}_n{block_n}_w{num_warps}"


def _make_provider(q, k, v, causal: bool, config: tuple[int, int, int]):
    block_m, block_n, num_warps = config

    def provider():
        return flash_attention_forward_configured(
            q,
            k,
            v,
            causal=causal,
            block_m=block_m,
            block_n=block_n,
            num_warps=num_warps,
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


def _tflops(sequence_length: int, latency_ms: float) -> float:
    flops = (
        4
        * BATCH_SIZE
        * NUM_HEADS
        * sequence_length
        * sequence_length
        * HEAD_DIM
    )
    return flops / latency_ms * 1e-9


def _write_csv(path: Path, rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(raw_rows: list[dict]) -> list[dict]:
    summary_rows = []
    for causal in (False, True):
        for sequence_length in SEQUENCE_LENGTHS:
            for config in CANDIDATES:
                name = _config_name(config)
                group = [
                    row
                    for row in raw_rows
                    if row["causal"] == causal
                    and row["sequence_length"] == sequence_length
                    and row["config"] == name
                ]
                p50_values = [float(row["p50_ms"]) for row in group]
                speedups = [float(row["baseline_speedup"]) for row in group]
                tflops_values = [float(row["tflops"]) for row in group]
                block_m, block_n, num_warps = config
                summary_rows.append(
                    {
                        "causal": causal,
                        "sequence_length": sequence_length,
                        "block_m": block_m,
                        "block_n": block_n,
                        "num_warps": num_warps,
                        "config": name,
                        "rounds": len(group),
                        "median_p50_ms": statistics.median(p50_values),
                        "min_p50_ms": min(p50_values),
                        "max_p50_ms": max(p50_values),
                        "median_tflops": statistics.median(tflops_values),
                        "median_baseline_speedup": statistics.median(speedups),
                        "min_baseline_speedup": min(speedups),
                        "max_baseline_speedup": max(speedups),
                    }
                )
    return summary_rows


def _plot(summary_rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    names = [_config_name(config) for config in CANDIDATES]
    x_positions = list(range(len(names)))

    for axis, causal in zip(axes, (False, True)):
        for sequence_length in SEQUENCE_LENGTHS:
            lookup = {
                row["config"]: row
                for row in summary_rows
                if row["causal"] == causal
                and row["sequence_length"] == sequence_length
            }
            medians = [lookup[name]["median_baseline_speedup"] for name in names]
            lower = [
                lookup[name]["median_baseline_speedup"]
                - lookup[name]["min_baseline_speedup"]
                for name in names
            ]
            upper = [
                lookup[name]["max_baseline_speedup"]
                - lookup[name]["median_baseline_speedup"]
                for name in names
            ]
            axis.errorbar(
                x_positions,
                medians,
                yerr=(lower, upper),
                marker="o",
                capsize=3,
                label=f"N={sequence_length}",
            )

        axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
        axis.set_title("Causal" if causal else "Non-causal")
        axis.set_xticks(x_positions, names, rotation=30, ha="right")
        axis.set_ylabel("Median speedup over per-round baseline")
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle("FlashAttention Final Configuration Selection (3 Rounds)")
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_final_selection() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark 需要 CUDA GPU")

    device = torch.device("cuda")
    reference = _load_naive_reference()
    raw_rows = []
    torch.manual_seed(0)

    print(
        f"B={BATCH_SIZE}, H={NUM_HEADS}, D={HEAD_DIM}, dtype={DTYPE}, "
        f"rounds={ROUNDS}, baseline={_config_name(BASELINE)}"
    )

    for causal in (False, True):
        for sequence_length in SEQUENCE_LENGTHS:
            shape = (BATCH_SIZE, NUM_HEADS, sequence_length, HEAD_DIM)
            q = torch.randn(shape, device=device, dtype=DTYPE)
            k = torch.randn(shape, device=device, dtype=DTYPE)
            v = torch.randn(shape, device=device, dtype=DTYPE)
            expected = reference(q, k, v, causal=causal)[0]
            providers = {
                config: _make_provider(q, k, v, causal, config)
                for config in CANDIDATES
            }

            # 先完成所有候选的 correctness/JIT，避免编译时间进入正式测量。
            for config, provider in providers.items():
                actual = provider()
                torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
                del actual
            torch.cuda.synchronize(device)

            for round_idx in range(ROUNDS):
                order = list(CANDIDATES)
                random.Random(
                    2026 + round_idx + sequence_length + 10000 * int(causal)
                ).shuffle(order)
                round_rows = []

                for config in order:
                    p20, p50, p80 = _bench(providers[config])
                    block_m, block_n, num_warps = config
                    round_rows.append(
                        {
                            "causal": causal,
                            "sequence_length": sequence_length,
                            "round": round_idx,
                            "measurement_order": order.index(config),
                            "block_m": block_m,
                            "block_n": block_n,
                            "num_warps": num_warps,
                            "config": _config_name(config),
                            "p20_ms": p20,
                            "p50_ms": p50,
                            "p80_ms": p80,
                            "tflops": _tflops(sequence_length, p50),
                            "baseline_speedup": 0.0,
                        }
                    )

                baseline_p50 = next(
                    float(row["p50_ms"])
                    for row in round_rows
                    if row["config"] == _config_name(BASELINE)
                )
                for row in round_rows:
                    row["baseline_speedup"] = baseline_p50 / float(row["p50_ms"])
                    print(
                        f"causal={causal:<5} N={sequence_length:<4d} "
                        f"round={round_idx} order={row['measurement_order']} "
                        f"{row['config']:<12} P50={float(row['p50_ms']):.4f}ms "
                        f"speedup={float(row['baseline_speedup']):.3f}x"
                    )
                raw_rows.extend(round_rows)

            del expected, q, k, v

    summary_rows = _summarize(raw_rows)
    _write_csv(RAW_CSV_PATH, raw_rows)
    _write_csv(SUMMARY_CSV_PATH, summary_rows)
    _plot(summary_rows)

    print("\n每个场景的最佳 median P50：")
    for causal in (False, True):
        for sequence_length in SEQUENCE_LENGTHS:
            candidates = [
                row
                for row in summary_rows
                if row["causal"] == causal
                and row["sequence_length"] == sequence_length
            ]
            best = min(candidates, key=lambda row: row["median_p50_ms"])
            print(
                f"causal={causal:<5} N={sequence_length:<4d} "
                f"best={best['config']} "
                f"P50={best['median_p50_ms']:.4f}ms "
                f"speedup={best['median_baseline_speedup']:.3f}x"
            )

    print(f"Raw CSV: {RAW_CSV_PATH}")
    print(f"Summary CSV: {SUMMARY_CSV_PATH}")
    print(f"Plot: {PLOT_PATH}")


if __name__ == "__main__":
    run_final_selection()
