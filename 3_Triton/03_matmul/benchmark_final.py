import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import triton
import triton.testing

if __package__:
    from .kernel import matmul, matmul_autotuned
else:
    from kernel import matmul, matmul_autotuned


DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE = torch.float16

WARMUP_MS = 100
REP_MS = 300
QUANTILES = [0.2, 0.5, 0.8]
RESULT_DIR = Path(__file__).with_name("benchmark_results")

MANUAL_CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": 64,
    "BLOCK_K": 64,
    "GROUP_SIZE_M": 2,
    "num_warps": 4,
}

SHAPES = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (512, 1024, 256),
    (1024, 512, 2048),
    (2048, 1024, 512),
]


def compute_tflops(M: int, N: int, K: int, latency_ms: float) -> float:
    operations = 2 * M * N * K
    return operations / latency_ms * 1e-9


def measure(fn):
    """返回 P20、P50、P80 延迟，单位为毫秒。"""
    return triton.testing.do_bench(
        fn,
        warmup=WARMUP_MS,
        rep=REP_MS,
        quantiles=QUANTILES,
    )


def benchmark_case(M: int, N: int, K: int) -> dict:
    a = torch.randn((M, K), device=DEVICE, dtype=DTYPE)
    b = torch.randn((K, N), device=DEVICE, dtype=DTYPE)

    # correctness 调用同时完成 JIT 编译和 autotune，不进入计时区间。
    expected = torch.matmul(a, b)
    manual_actual = matmul(a, b, **MANUAL_CONFIG)
    autotuned_actual = matmul_autotuned(a, b)

    torch.testing.assert_close(
        manual_actual,
        expected,
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        autotuned_actual,
        expected,
        atol=2e-2,
        rtol=2e-2,
    )
    del expected, manual_actual, autotuned_actual

    torch_p20, torch_p50, torch_p80 = measure(
        lambda: torch.matmul(a, b)
    )
    manual_p20, manual_p50, manual_p80 = measure(
        lambda: matmul(a, b, **MANUAL_CONFIG)
    )
    autotuned_p20, autotuned_p50, autotuned_p80 = measure(
        lambda: matmul_autotuned(a, b)
    )

    return {
        "M": M,
        "N": N,
        "K": K,
        "torch_p20_ms": torch_p20,
        "torch_p50_ms": torch_p50,
        "torch_p80_ms": torch_p80,
        "manual_p20_ms": manual_p20,
        "manual_p50_ms": manual_p50,
        "manual_p80_ms": manual_p80,
        "autotuned_p20_ms": autotuned_p20,
        "autotuned_p50_ms": autotuned_p50,
        "autotuned_p80_ms": autotuned_p80,
        "torch_tflops": compute_tflops(M, N, K, torch_p50),
        "manual_tflops": compute_tflops(M, N, K, manual_p50),
        "autotuned_tflops": compute_tflops(M, N, K, autotuned_p50),
        "manual_speedup": torch_p50 / manual_p50,
        "autotuned_speedup": torch_p50 / autotuned_p50,
    }


def print_environment():
    properties = torch.cuda.get_device_properties(DEVICE)

    print("\n=== Environment ===")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Triton:       {triton.__version__}")
    print(f"CUDA:         {torch.version.cuda}")
    print(f"GPU:          {properties.name}")
    print(f"dtype:        {DTYPE}")
    print(f"manual:       {MANUAL_CONFIG}")
    print(f"warmup/rep:   {WARMUP_MS}/{REP_MS} ms")


def print_results(rows: list[dict]):
    print("\n=== Final MatMul Benchmark ===")
    print(
        f"{'M':>6} {'N':>6} {'K':>6} "
        f"{'Torch ms':>10} {'Manual ms':>10} {'Auto ms':>10} "
        f"{'Torch TF':>10} {'Manual TF':>10} {'Auto TF':>10} "
        f"{'Manual x':>9} {'Auto x':>9}"
    )
    for row in rows:
        print(
            f"{row['M']:>6} {row['N']:>6} {row['K']:>6} "
            f"{row['torch_p50_ms']:>10.4f} "
            f"{row['manual_p50_ms']:>10.4f} "
            f"{row['autotuned_p50_ms']:>10.4f} "
            f"{row['torch_tflops']:>10.2f} "
            f"{row['manual_tflops']:>10.2f} "
            f"{row['autotuned_tflops']:>10.2f} "
            f"{row['manual_speedup']:>9.3f} "
            f"{row['autotuned_speedup']:>9.3f}"
        )
    print("Speedup > 1 表示对应 Triton 实现比 PyTorch 更快。")


def save_csv(rows: list[dict]):
    path = RESULT_DIR / "matmul-final.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV 已保存：{path}")


def save_plot(rows: list[dict]):
    positions = list(range(len(rows)))
    labels = [f"{row['M']}x{row['N']}x{row['K']}" for row in rows]
    width = 0.26

    figure, (latency_axis, tflops_axis) = plt.subplots(
        1,
        2,
        figsize=(16, 6),
    )

    latency_axis.plot(
        positions,
        [row["torch_p50_ms"] for row in rows],
        marker="o",
        label="PyTorch",
    )
    latency_axis.plot(
        positions,
        [row["manual_p50_ms"] for row in rows],
        marker="o",
        label="Triton manual",
    )
    latency_axis.plot(
        positions,
        [row["autotuned_p50_ms"] for row in rows],
        marker="o",
        label="Triton autotuned",
    )
    latency_axis.set_yscale("log")
    latency_axis.set_ylabel("P50 latency (ms, log scale)")
    latency_axis.set_title("Latency: lower is better")
    latency_axis.grid(True, alpha=0.3)
    latency_axis.legend()

    torch_bars = tflops_axis.bar(
        [position - width for position in positions],
        [row["torch_tflops"] for row in rows],
        width,
        label="PyTorch",
    )
    manual_bars = tflops_axis.bar(
        positions,
        [row["manual_tflops"] for row in rows],
        width,
        label="Triton manual",
    )
    autotuned_bars = tflops_axis.bar(
        [position + width for position in positions],
        [row["autotuned_tflops"] for row in rows],
        width,
        label="Triton autotuned",
    )
    tflops_axis.bar_label(torch_bars, fmt="%.1f", padding=3, fontsize=7)
    tflops_axis.bar_label(manual_bars, fmt="%.1f", padding=3, fontsize=7)
    tflops_axis.bar_label(autotuned_bars, fmt="%.1f", padding=3, fontsize=7)
    tflops_axis.set_ylabel("TFLOPS")
    tflops_axis.set_title("Throughput: higher is better")
    tflops_axis.grid(True, axis="y", alpha=0.3)
    tflops_axis.legend()

    for axis in (latency_axis, tflops_axis):
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.set_xlabel("Matrix shape (M x N x K)")

    figure.suptitle("Final Triton MatMul: manual vs autotuned vs PyTorch")
    figure.tight_layout()

    path = RESULT_DIR / "matmul-final.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"图像已保存：{path}")


if __name__ == "__main__":
    torch.manual_seed(0)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print_environment()

    results = [
        benchmark_case(M, N, K)
        for M, N, K in SHAPES
    ]

    print_results(results)
    save_csv(results)
    save_plot(results)
