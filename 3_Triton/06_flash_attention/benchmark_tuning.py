import csv
import importlib.util
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
WARMUP_MS = 100
REP_MS = 300
QUANTILES = (0.2, 0.5, 0.8)

BASELINE = (32, 32, 4)
TILE_CONFIGS = (
    (16, 32, 4),
    (32, 16, 4),
    BASELINE,
    (32, 64, 4),
    (64, 32, 4),
)
WARP_CONFIGS = (
    (32, 32, 2),
    BASELINE,
    (32, 32, 8),
)
ALL_CONFIGS = tuple(dict.fromkeys(TILE_CONFIGS + WARP_CONFIGS))

RESULTS_DIR = Path(__file__).resolve().parent / "benchmark_results"
CSV_PATH = RESULTS_DIR / "flash-attention-tuning.csv"
PLOT_PATH = RESULTS_DIR / "flash-attention-tuning.png"


@lru_cache(maxsize=1)
def _load_naive_attention_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "05_attention_naive"
        / "kernel.py"
    )
    spec = importlib.util.spec_from_file_location(
        "attention_naive_kernel_for_flash_tuning",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 05 实现：{module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config_name(config: tuple[int, int, int]) -> str:
    block_m, block_n, num_warps = config
    return f"m{block_m}_n{block_n}_w{num_warps}"


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


def _write_csv(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "causal",
        "sequence_length",
        "block_m",
        "block_n",
        "num_warps",
        "config",
        "status",
        "error",
        "p20_ms",
        "p50_ms",
        "p80_ms",
        "tflops",
        "baseline_speedup",
    )
    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_experiment(
    axis,
    rows: list[dict],
    causal: bool,
    configs: tuple[tuple[int, int, int], ...],
    title: str,
) -> None:
    names = [_config_name(config) for config in configs]
    for sequence_length in SEQUENCE_LENGTHS:
        lookup = {
            row["config"]: row
            for row in rows
            if row["status"] == "ok"
            and row["causal"] == causal
            and row["sequence_length"] == sequence_length
        }
        speedups = [
            lookup[name]["baseline_speedup"] if name in lookup else float("nan")
            for name in names
        ]
        axis.plot(names, speedups, marker="o", label=f"N={sequence_length}")

    axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    axis.set_title(f"{title} - {'Causal' if causal else 'Non-causal'}")
    axis.set_ylabel("Speedup over m32_n32_w4")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(True, alpha=0.3)
    axis.legend()


def _plot(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    _plot_experiment(axes[0, 0], rows, False, TILE_CONFIGS, "Tile sweep")
    _plot_experiment(axes[0, 1], rows, True, TILE_CONFIGS, "Tile sweep")
    _plot_experiment(axes[1, 0], rows, False, WARP_CONFIGS, "Warp sweep")
    _plot_experiment(axes[1, 1], rows, True, WARP_CONFIGS, "Warp sweep")
    figure.suptitle("FlashAttention Forward Controlled Tuning")
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_tuning_benchmark() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark 需要 CUDA GPU")

    device = torch.device("cuda")
    naive_reference = _load_naive_attention_module().naive_attention_reference
    rows = []
    torch.manual_seed(0)

    print(
        f"B={BATCH_SIZE}, H={NUM_HEADS}, D={HEAD_DIM}, dtype={DTYPE}, "
        f"baseline={_config_name(BASELINE)}"
    )
    print("causal N    config       P20/P50/P80(ms)       TFLOPS  vs baseline")

    for causal in (False, True):
        for sequence_length in SEQUENCE_LENGTHS:
            shape = (BATCH_SIZE, NUM_HEADS, sequence_length, HEAD_DIM)
            q = torch.randn(shape, device=device, dtype=DTYPE)
            k = torch.randn(shape, device=device, dtype=DTYPE)
            v = torch.randn(shape, device=device, dtype=DTYPE)
            expected = naive_reference(q, k, v, causal=causal)[0]
            shape_rows = []

            for block_m, block_n, num_warps in ALL_CONFIGS:
                config = (block_m, block_n, num_warps)
                name = _config_name(config)

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

                row = {
                    "causal": causal,
                    "sequence_length": sequence_length,
                    "block_m": block_m,
                    "block_n": block_n,
                    "num_warps": num_warps,
                    "config": name,
                    "status": "ok",
                    "error": "",
                    "p20_ms": "",
                    "p50_ms": "",
                    "p80_ms": "",
                    "tflops": "",
                    "baseline_speedup": "",
                }
                try:
                    actual = provider()
                    torch.testing.assert_close(
                        actual,
                        expected,
                        atol=2e-2,
                        rtol=2e-2,
                    )
                    torch.cuda.synchronize(device)
                    del actual

                    p20, p50, p80 = _bench(provider)
                    row.update(
                        {
                            "p20_ms": p20,
                            "p50_ms": p50,
                            "p80_ms": p80,
                            "tflops": _tflops(sequence_length, p50),
                        }
                    )
                except Exception as error:  # 失败候选也写入 CSV，保留实验记录。
                    row["status"] = "failed"
                    row["error"] = f"{type(error).__name__}: {error}"

                shape_rows.append(row)

            baseline_row = next(
                row
                for row in shape_rows
                if row["config"] == _config_name(BASELINE)
            )
            if baseline_row["status"] != "ok":
                raise RuntimeError(
                    f"baseline 在 causal={causal}, N={sequence_length} 失败："
                    f"{baseline_row['error']}"
                )

            baseline_p50 = float(baseline_row["p50_ms"])
            for row in shape_rows:
                if row["status"] == "ok":
                    row["baseline_speedup"] = baseline_p50 / float(row["p50_ms"])
                    print(
                        f"{str(causal):<7} {sequence_length:<4d} "
                        f"{row['config']:<12} "
                        f"{float(row['p20_ms']):.4f}/"
                        f"{float(row['p50_ms']):.4f}/"
                        f"{float(row['p80_ms']):.4f} "
                        f"{float(row['tflops']):>8.3f} "
                        f"{float(row['baseline_speedup']):>10.3f}x"
                    )
                else:
                    print(
                        f"{str(causal):<7} {sequence_length:<4d} "
                        f"{row['config']:<12} FAILED: {row['error']}"
                    )
            rows.extend(shape_rows)
            del expected, q, k, v

    _write_csv(rows)
    _plot(rows)
    print(f"CSV: {CSV_PATH}")
    print(f"Plot: {PLOT_PATH}")


if __name__ == "__main__":
    run_tuning_benchmark()
