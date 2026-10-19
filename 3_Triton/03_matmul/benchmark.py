import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import triton
import triton.testing

if __package__:
    from .kernel import matmul
else:
    from kernel import matmul


DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE = torch.float16

WARMUP_MS = 100
REP_MS = 300
QUANTILES = [0.2, 0.5, 0.8]
RESULT_DIR = Path(__file__).with_name("benchmark_results")

BLOCK_CONFIGS = [
    {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "num_warps": 4, "GROUP_SIZE_M": 1},
    {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "GROUP_SIZE_M": 1},
    {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "GROUP_SIZE_M": 1},
    {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "GROUP_SIZE_M": 1},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 8, "GROUP_SIZE_M": 1},
]

GROUP_SIZES = [1, 2, 4, 8, 16]
BLOCK_K_SIZES = [16, 32, 64, 128]
NUM_WARPS_VALUES = [2, 4, 8]


def compute_tflops(M: int, N: int, K: int, latency_ms: float) -> float:
    # MatMul 总计算量约为 2*M*N*K 次浮点运算。
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
    a = torch.randn(
        (M, K),
        device=DEVICE,
        dtype=DTYPE,
    )
    b = torch.randn(
        (K, N),
        device=DEVICE,
        dtype=DTYPE,
    )

    # 计时前先验证一次，同时触发 Triton JIT 编译。
    expected = torch.matmul(a, b)
    actual = matmul(a, b, GROUP_SIZE_M=1)
    torch.testing.assert_close(
        actual,
        expected,
        atol=2e-2,
        rtol=2e-2,
    )
    del expected, actual

    torch_p20, torch_p50, torch_p80 = measure(
        lambda: torch.matmul(a, b)
    )
    triton_p20, triton_p50, triton_p80 = measure(
        lambda: matmul(a, b, GROUP_SIZE_M=1)
    )

    return {
        "M": M,
        "N": N,
        "K": K,
        "torch_p20_ms": torch_p20,
        "torch_p50_ms": torch_p50,
        "torch_p80_ms": torch_p80,
        "triton_p20_ms": triton_p20,
        "triton_p50_ms": triton_p50,
        "triton_p80_ms": triton_p80,
        "torch_tflops": compute_tflops(M, N, K, torch_p50),
        "triton_tflops": compute_tflops(M, N, K, triton_p50),
        "speedup": torch_p50 / triton_p50,
    }


def print_environment():
    properties = torch.cuda.get_device_properties(DEVICE)

    print("\n=== Environment ===")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Triton:       {triton.__version__}")
    print(f"CUDA:         {torch.version.cuda}")
    print(f"GPU:          {properties.name}")
    print(f"dtype:        {DTYPE}")
    print("block:        32 x 32 x 32")
    print("group size:   1 (ungrouped baseline)")
    print(f"warmup/rep:   {WARMUP_MS}/{REP_MS} ms")


def print_results(rows: list[dict]):
    print("\n=== MatMul Benchmark ===")
    print(
        f"{'M':>6} {'N':>6} {'K':>6} "
        f"{'Torch ms':>10} {'Triton ms':>10} "
        f"{'Torch TFLOPS':>13} {'Triton TFLOPS':>14} "
        f"{'Speedup':>9}"
    )

    for row in rows:
        print(
            f"{row['M']:>6} "
            f"{row['N']:>6} "
            f"{row['K']:>6} "
            f"{row['torch_p50_ms']:>10.4f} "
            f"{row['triton_p50_ms']:>10.4f} "
            f"{row['torch_tflops']:>13.2f} "
            f"{row['triton_tflops']:>14.2f} "
            f"{row['speedup']:>9.3f}"
        )

    print("\nSpeedup > 1 表示 Triton 更快。")


def save_csv(rows: list[dict]):
    path = RESULT_DIR / "matmul-naive.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV 已保存：{path}")


def save_plot(rows: list[dict]):
    positions = list(range(len(rows)))
    labels = [f"{row['M']}x{row['N']}x{row['K']}" for row in rows]

    torch_latency = [row["torch_p50_ms"] for row in rows]
    triton_latency = [row["triton_p50_ms"] for row in rows]
    torch_tflops = [row["torch_tflops"] for row in rows]
    triton_tflops = [row["triton_tflops"] for row in rows]

    figure, (latency_axis, tflops_axis) = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    latency_axis.plot(
        positions,
        torch_latency,
        marker="o",
        label="PyTorch",
    )
    latency_axis.plot(
        positions,
        triton_latency,
        marker="o",
        label="Triton naive",
    )
    latency_axis.set_yscale("log")
    latency_axis.set_ylabel("P50 latency (ms, log scale)")
    latency_axis.set_title("Latency: lower is better")
    latency_axis.grid(True, alpha=0.3)
    latency_axis.legend()

    width = 0.36
    torch_bars = tflops_axis.bar(
        [position - width / 2 for position in positions],
        torch_tflops,
        width,
        label="PyTorch",
    )
    triton_bars = tflops_axis.bar(
        [position + width / 2 for position in positions],
        triton_tflops,
        width,
        label="Triton naive",
    )
    tflops_axis.bar_label(torch_bars, fmt="%.2f", padding=3, fontsize=8)
    tflops_axis.bar_label(triton_bars, fmt="%.2f", padding=3, fontsize=8)
    tflops_axis.set_ylabel("TFLOPS")
    tflops_axis.set_title("Throughput: higher is better")
    tflops_axis.grid(True, axis="y", alpha=0.3)
    tflops_axis.legend()

    for axis in (latency_axis, tflops_axis):
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=30, ha="right")
        axis.set_xlabel("Matrix shape (M x N x K)")

    figure.suptitle("Naive Triton MatMul vs PyTorch")
    figure.tight_layout()

    path = RESULT_DIR / "matmul-naive.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"图像已保存：{path}")


def config_label(config: dict) -> str:
    return (
        f"{config['BLOCK_M']}x{config['BLOCK_N']}x{config['BLOCK_K']}"
        f"\nw{config['num_warps']}"
    )


def shape_label(rows: list[dict]) -> str:
    first = rows[0]
    return f"{first['M']} x {first['N']} x {first['K']}"


def shape_slug(rows: list[dict]) -> str:
    first = rows[0]
    if first["M"] == first["N"] == first["K"]:
        return str(first["M"])
    return f"{first['M']}x{first['N']}x{first['K']}"


def run_block_sweep(M: int, N: int, K: int) -> list[dict]:
    a = torch.randn((M, K), device=DEVICE, dtype=DTYPE)
    b = torch.randn((K, N), device=DEVICE, dtype=DTYPE)
    expected = torch.matmul(a, b)

    torch_p20, torch_p50, torch_p80 = measure(
        lambda: torch.matmul(a, b)
    )
    torch_tflops = compute_tflops(M, N, K, torch_p50)

    rows = []
    for config in BLOCK_CONFIGS:
        actual = matmul(a, b, **config)
        torch.testing.assert_close(
            actual,
            expected,
            atol=2e-2,
            rtol=2e-2,
        )
        del actual

        triton_p20, triton_p50, triton_p80 = measure(
            lambda: matmul(a, b, **config)
        )

        rows.append(
            {
                "M": M,
                "N": N,
                "K": K,
                **config,
                "torch_p20_ms": torch_p20,
                "torch_p50_ms": torch_p50,
                "torch_p80_ms": torch_p80,
                "triton_p20_ms": triton_p20,
                "triton_p50_ms": triton_p50,
                "triton_p80_ms": triton_p80,
                "torch_tflops": torch_tflops,
                "triton_tflops": compute_tflops(M, N, K, triton_p50),
                "speedup": torch_p50 / triton_p50,
            }
        )

    return rows


def print_block_sweep(rows: list[dict]):
    print(f"\n=== Block-size sweep: {shape_label(rows)} ===")
    print(
        f"{'BM':>5} {'BN':>5} {'BK':>5} {'Warps':>6} "
        f"{'Triton ms':>10} {'TFLOPS':>10} {'vs Torch':>10}"
    )
    for row in rows:
        print(
            f"{row['BLOCK_M']:>5} "
            f"{row['BLOCK_N']:>5} "
            f"{row['BLOCK_K']:>5} "
            f"{row['num_warps']:>6} "
            f"{row['triton_p50_ms']:>10.4f} "
            f"{row['triton_tflops']:>10.2f} "
            f"{row['speedup']:>10.3f}"
        )
    print(
        f"PyTorch baseline: {rows[0]['torch_p50_ms']:.4f} ms, "
        f"{rows[0]['torch_tflops']:.2f} TFLOPS"
    )


def save_block_sweep_csv(rows: list[dict]):
    path = RESULT_DIR / f"matmul-block-sweep-{shape_slug(rows)}.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV 已保存：{path}")


def save_block_sweep_plot(rows: list[dict]):
    positions = list(range(len(rows)))
    labels = [config_label(row) for row in rows]
    latency = [row["triton_p50_ms"] for row in rows]
    tflops = [row["triton_tflops"] for row in rows]
    torch_latency = rows[0]["torch_p50_ms"]
    torch_tflops = rows[0]["torch_tflops"]

    figure, (latency_axis, tflops_axis) = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    latency_bars = latency_axis.bar(positions, latency)
    latency_axis.axhline(
        torch_latency,
        color="black",
        linestyle="--",
        label=f"PyTorch: {torch_latency:.4f} ms",
    )
    latency_axis.bar_label(latency_bars, fmt="%.4f", padding=3, fontsize=8)
    latency_axis.set_ylabel("P50 latency (ms)")
    latency_axis.set_title("Latency: lower is better")
    latency_axis.grid(True, axis="y", alpha=0.3)
    latency_axis.legend()

    tflops_bars = tflops_axis.bar(positions, tflops, color="tab:orange")
    tflops_axis.axhline(
        torch_tflops,
        color="black",
        linestyle="--",
        label=f"PyTorch: {torch_tflops:.2f} TFLOPS",
    )
    tflops_axis.bar_label(tflops_bars, fmt="%.2f", padding=3, fontsize=8)
    tflops_axis.set_ylabel("TFLOPS")
    tflops_axis.set_title("Throughput: higher is better")
    tflops_axis.grid(True, axis="y", alpha=0.3)
    tflops_axis.legend()

    for axis in (latency_axis, tflops_axis):
        axis.set_xticks(positions)
        axis.set_xticklabels(labels)
        axis.set_xlabel("BLOCK_M x BLOCK_N x BLOCK_K / num_warps")

    figure.suptitle(f"Triton MatMul block-size sweep: {shape_label(rows)}")
    figure.tight_layout()

    path = RESULT_DIR / f"matmul-block-sweep-{shape_slug(rows)}.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"图像已保存：{path}")


def run_group_sweep(M: int, N: int, K: int) -> list[dict]:
    config = {
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "BLOCK_K": 32,
        "num_warps": 4,
    }
    a = torch.randn((M, K), device=DEVICE, dtype=DTYPE)
    b = torch.randn((K, N), device=DEVICE, dtype=DTYPE)
    expected = torch.matmul(a, b)

    torch_p20, torch_p50, torch_p80 = measure(
        lambda: torch.matmul(a, b)
    )
    torch_tflops = compute_tflops(M, N, K, torch_p50)

    rows = []
    for group_size_m in GROUP_SIZES:
        actual = matmul(
            a,
            b,
            **config,
            GROUP_SIZE_M=group_size_m,
        )
        torch.testing.assert_close(
            actual,
            expected,
            atol=2e-2,
            rtol=2e-2,
        )
        del actual

        triton_p20, triton_p50, triton_p80 = measure(
            lambda: matmul(
                a,
                b,
                **config,
                GROUP_SIZE_M=group_size_m,
            )
        )

        rows.append(
            {
                "M": M,
                "N": N,
                "K": K,
                **config,
                "GROUP_SIZE_M": group_size_m,
                "torch_p20_ms": torch_p20,
                "torch_p50_ms": torch_p50,
                "torch_p80_ms": torch_p80,
                "triton_p20_ms": triton_p20,
                "triton_p50_ms": triton_p50,
                "triton_p80_ms": triton_p80,
                "torch_tflops": torch_tflops,
                "triton_tflops": compute_tflops(M, N, K, triton_p50),
                "speedup_vs_torch": torch_p50 / triton_p50,
            }
        )

    group_one_latency = rows[0]["triton_p50_ms"]
    for row in rows:
        row["speedup_vs_group1"] = group_one_latency / row["triton_p50_ms"]

    return rows


def print_group_sweep(rows: list[dict]):
    print(f"\n=== GROUP_SIZE_M sweep: {shape_label(rows)} ===")
    print(
        f"{'Group':>7} {'P50 ms':>10} {'TFLOPS':>10} "
        f"{'vs Group1':>11} {'vs Torch':>10}"
    )
    for row in rows:
        print(
            f"{row['GROUP_SIZE_M']:>7} "
            f"{row['triton_p50_ms']:>10.4f} "
            f"{row['triton_tflops']:>10.2f} "
            f"{row['speedup_vs_group1']:>11.3f} "
            f"{row['speedup_vs_torch']:>10.3f}"
        )
    print(
        f"PyTorch baseline: {rows[0]['torch_p50_ms']:.4f} ms, "
        f"{rows[0]['torch_tflops']:.2f} TFLOPS"
    )


def save_group_sweep_csv(rows: list[dict]):
    path = RESULT_DIR / f"matmul-group-sweep-{shape_slug(rows)}.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV 已保存：{path}")


def save_group_sweep_plot(rows: list[dict]):
    positions = list(range(len(rows)))
    labels = [str(row["GROUP_SIZE_M"]) for row in rows]
    latency = [row["triton_p50_ms"] for row in rows]
    tflops = [row["triton_tflops"] for row in rows]
    torch_latency = rows[0]["torch_p50_ms"]
    torch_tflops = rows[0]["torch_tflops"]

    figure, (latency_axis, tflops_axis) = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    latency_bars = latency_axis.bar(positions, latency)
    latency_axis.axhline(
        torch_latency,
        color="black",
        linestyle="--",
        label=f"PyTorch: {torch_latency:.4f} ms",
    )
    latency_axis.bar_label(latency_bars, fmt="%.4f", padding=3, fontsize=8)
    latency_axis.set_ylabel("P50 latency (ms)")
    latency_axis.set_title("Latency: lower is better")
    latency_axis.grid(True, axis="y", alpha=0.3)
    latency_axis.legend()

    tflops_bars = tflops_axis.bar(positions, tflops, color="tab:orange")
    tflops_axis.axhline(
        torch_tflops,
        color="black",
        linestyle="--",
        label=f"PyTorch: {torch_tflops:.2f} TFLOPS",
    )
    tflops_axis.bar_label(tflops_bars, fmt="%.2f", padding=3, fontsize=8)
    tflops_axis.set_ylabel("TFLOPS")
    tflops_axis.set_title("Throughput: higher is better")
    tflops_axis.grid(True, axis="y", alpha=0.3)
    tflops_axis.legend()

    for axis in (latency_axis, tflops_axis):
        axis.set_xticks(positions)
        axis.set_xticklabels(labels)
        axis.set_xlabel("GROUP_SIZE_M")

    figure.suptitle(
        f"Triton MatMul grouped ordering: {shape_label(rows)} "
        "(BM=128, BN=64, BK=32, w4)"
    )
    figure.tight_layout()

    path = RESULT_DIR / f"matmul-group-sweep-{shape_slug(rows)}.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"图像已保存：{path}")


def run_block_k_sweep(M: int, N: int, K: int) -> list[dict]:
    base_config = {
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "num_warps": 4,
        "GROUP_SIZE_M": 2,
    }
    a = torch.randn((M, K), device=DEVICE, dtype=DTYPE)
    b = torch.randn((K, N), device=DEVICE, dtype=DTYPE)
    expected = torch.matmul(a, b)

    torch_p20, torch_p50, torch_p80 = measure(
        lambda: torch.matmul(a, b)
    )
    torch_tflops = compute_tflops(M, N, K, torch_p50)

    rows = []
    for block_k in BLOCK_K_SIZES:
        actual = matmul(
            a,
            b,
            **base_config,
            BLOCK_K=block_k,
        )
        torch.testing.assert_close(
            actual,
            expected,
            atol=2e-2,
            rtol=2e-2,
        )
        del actual

        triton_p20, triton_p50, triton_p80 = measure(
            lambda: matmul(
                a,
                b,
                **base_config,
                BLOCK_K=block_k,
            )
        )

        rows.append(
            {
                "M": M,
                "N": N,
                "K": K,
                **base_config,
                "BLOCK_K": block_k,
                "torch_p20_ms": torch_p20,
                "torch_p50_ms": torch_p50,
                "torch_p80_ms": torch_p80,
                "triton_p20_ms": triton_p20,
                "triton_p50_ms": triton_p50,
                "triton_p80_ms": triton_p80,
                "torch_tflops": torch_tflops,
                "triton_tflops": compute_tflops(M, N, K, triton_p50),
                "speedup_vs_torch": torch_p50 / triton_p50,
            }
        )

    block_k_32_latency = next(
        row["triton_p50_ms"]
        for row in rows
        if row["BLOCK_K"] == 32
    )
    for row in rows:
        row["speedup_vs_bk32"] = block_k_32_latency / row["triton_p50_ms"]

    return rows


def print_block_k_sweep(rows: list[dict]):
    print(f"\n=== BLOCK_K sweep: {shape_label(rows)} ===")
    print(
        f"{'BK':>5} {'K loops':>8} {'P50 ms':>10} {'TFLOPS':>10} "
        f"{'vs BK32':>9} {'vs Torch':>10}"
    )
    for row in rows:
        print(
            f"{row['BLOCK_K']:>5} "
            f"{triton.cdiv(row['K'], row['BLOCK_K']):>8} "
            f"{row['triton_p50_ms']:>10.4f} "
            f"{row['triton_tflops']:>10.2f} "
            f"{row['speedup_vs_bk32']:>9.3f} "
            f"{row['speedup_vs_torch']:>10.3f}"
        )
    print(
        f"PyTorch baseline: {rows[0]['torch_p50_ms']:.4f} ms, "
        f"{rows[0]['torch_tflops']:.2f} TFLOPS"
    )


def save_block_k_sweep_csv(rows: list[dict]):
    path = RESULT_DIR / f"matmul-block-k-sweep-{shape_slug(rows)}.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV 已保存：{path}")


def save_block_k_sweep_plot(rows: list[dict]):
    positions = list(range(len(rows)))
    labels = [str(row["BLOCK_K"]) for row in rows]
    latency = [row["triton_p50_ms"] for row in rows]
    tflops = [row["triton_tflops"] for row in rows]
    torch_latency = rows[0]["torch_p50_ms"]
    torch_tflops = rows[0]["torch_tflops"]

    figure, (latency_axis, tflops_axis) = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    latency_bars = latency_axis.bar(positions, latency)
    latency_axis.axhline(
        torch_latency,
        color="black",
        linestyle="--",
        label=f"PyTorch: {torch_latency:.4f} ms",
    )
    latency_axis.bar_label(latency_bars, fmt="%.4f", padding=3, fontsize=8)
    latency_axis.set_ylabel("P50 latency (ms)")
    latency_axis.set_title("Latency: lower is better")
    latency_axis.grid(True, axis="y", alpha=0.3)
    latency_axis.legend()

    tflops_bars = tflops_axis.bar(positions, tflops, color="tab:orange")
    tflops_axis.axhline(
        torch_tflops,
        color="black",
        linestyle="--",
        label=f"PyTorch: {torch_tflops:.2f} TFLOPS",
    )
    tflops_axis.bar_label(tflops_bars, fmt="%.2f", padding=3, fontsize=8)
    tflops_axis.set_ylabel("TFLOPS")
    tflops_axis.set_title("Throughput: higher is better")
    tflops_axis.grid(True, axis="y", alpha=0.3)
    tflops_axis.legend()

    for axis in (latency_axis, tflops_axis):
        axis.set_xticks(positions)
        axis.set_xticklabels(labels)
        axis.set_xlabel("BLOCK_K")

    figure.suptitle(
        f"Triton MatMul BLOCK_K sweep: {shape_label(rows)} "
        "(BM=128, BN=64, w4, group=2)"
    )
    figure.tight_layout()

    path = RESULT_DIR / f"matmul-block-k-sweep-{shape_slug(rows)}.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"图像已保存：{path}")


def run_num_warps_sweep(M: int, N: int, K: int) -> list[dict]:
    base_config = {
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "BLOCK_K": 64,
        "GROUP_SIZE_M": 2,
    }
    a = torch.randn((M, K), device=DEVICE, dtype=DTYPE)
    b = torch.randn((K, N), device=DEVICE, dtype=DTYPE)
    expected = torch.matmul(a, b)

    torch_p20, torch_p50, torch_p80 = measure(
        lambda: torch.matmul(a, b)
    )
    torch_tflops = compute_tflops(M, N, K, torch_p50)

    rows = []
    for num_warps in NUM_WARPS_VALUES:
        actual = matmul(
            a,
            b,
            **base_config,
            num_warps=num_warps,
        )
        torch.testing.assert_close(
            actual,
            expected,
            atol=2e-2,
            rtol=2e-2,
        )
        del actual

        triton_p20, triton_p50, triton_p80 = measure(
            lambda: matmul(
                a,
                b,
                **base_config,
                num_warps=num_warps,
            )
        )

        rows.append(
            {
                "M": M,
                "N": N,
                "K": K,
                **base_config,
                "num_warps": num_warps,
                "torch_p20_ms": torch_p20,
                "torch_p50_ms": torch_p50,
                "torch_p80_ms": torch_p80,
                "triton_p20_ms": triton_p20,
                "triton_p50_ms": triton_p50,
                "triton_p80_ms": triton_p80,
                "torch_tflops": torch_tflops,
                "triton_tflops": compute_tflops(M, N, K, triton_p50),
                "speedup_vs_torch": torch_p50 / triton_p50,
            }
        )

    four_warp_latency = next(
        row["triton_p50_ms"]
        for row in rows
        if row["num_warps"] == 4
    )
    for row in rows:
        row["speedup_vs_w4"] = four_warp_latency / row["triton_p50_ms"]

    return rows


def print_num_warps_sweep(rows: list[dict]):
    print(f"\n=== num_warps sweep: {shape_label(rows)} ===")
    print(
        f"{'Warps':>7} {'Threads':>8} {'P50 ms':>10} {'TFLOPS':>10} "
        f"{'vs w4':>8} {'vs Torch':>10}"
    )
    for row in rows:
        print(
            f"{row['num_warps']:>7} "
            f"{row['num_warps'] * 32:>8} "
            f"{row['triton_p50_ms']:>10.4f} "
            f"{row['triton_tflops']:>10.2f} "
            f"{row['speedup_vs_w4']:>8.3f} "
            f"{row['speedup_vs_torch']:>10.3f}"
        )
    print(
        f"PyTorch baseline: {rows[0]['torch_p50_ms']:.4f} ms, "
        f"{rows[0]['torch_tflops']:.2f} TFLOPS"
    )


def save_num_warps_sweep_csv(rows: list[dict]):
    path = RESULT_DIR / f"matmul-num-warps-sweep-{shape_slug(rows)}.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV 已保存：{path}")


def save_num_warps_sweep_plot(rows: list[dict]):
    positions = list(range(len(rows)))
    labels = [str(row["num_warps"]) for row in rows]
    latency = [row["triton_p50_ms"] for row in rows]
    tflops = [row["triton_tflops"] for row in rows]
    torch_latency = rows[0]["torch_p50_ms"]
    torch_tflops = rows[0]["torch_tflops"]

    figure, (latency_axis, tflops_axis) = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    latency_bars = latency_axis.bar(positions, latency)
    latency_axis.axhline(
        torch_latency,
        color="black",
        linestyle="--",
        label=f"PyTorch: {torch_latency:.4f} ms",
    )
    latency_axis.bar_label(latency_bars, fmt="%.4f", padding=3, fontsize=8)
    latency_axis.set_ylabel("P50 latency (ms)")
    latency_axis.set_title("Latency: lower is better")
    latency_axis.grid(True, axis="y", alpha=0.3)
    latency_axis.legend()

    tflops_bars = tflops_axis.bar(positions, tflops, color="tab:orange")
    tflops_axis.axhline(
        torch_tflops,
        color="black",
        linestyle="--",
        label=f"PyTorch: {torch_tflops:.2f} TFLOPS",
    )
    tflops_axis.bar_label(tflops_bars, fmt="%.2f", padding=3, fontsize=8)
    tflops_axis.set_ylabel("TFLOPS")
    tflops_axis.set_title("Throughput: higher is better")
    tflops_axis.grid(True, axis="y", alpha=0.3)
    tflops_axis.legend()

    for axis in (latency_axis, tflops_axis):
        axis.set_xticks(positions)
        axis.set_xticklabels(labels)
        axis.set_xlabel("num_warps")

    figure.suptitle(
        f"Triton MatMul num_warps sweep: {shape_label(rows)} "
        "(BM=128, BN=64, BK=64, group=2)"
    )
    figure.tight_layout()

    path = RESULT_DIR / f"matmul-num-warps-sweep-{shape_slug(rows)}.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"图像已保存：{path}")


if __name__ == "__main__":
    torch.manual_seed(0)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print_environment()

    shapes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]

    results = [
        benchmark_case(M, N, K)
        for M, N, K in shapes
    ]

    print_results(results)
    save_csv(results)
    save_plot(results)

    for M, N, K in [
        (512, 512, 512),
        (2048, 2048, 2048),
    ]:
        block_sweep_rows = run_block_sweep(M, N, K)
        print_block_sweep(block_sweep_rows)
        save_block_sweep_csv(block_sweep_rows)
        save_block_sweep_plot(block_sweep_rows)

    group_sweep_rows = run_group_sweep(2048, 2048, 2048)
    print_group_sweep(group_sweep_rows)
    save_group_sweep_csv(group_sweep_rows)
    save_group_sweep_plot(group_sweep_rows)

    block_k_sweep_rows = run_block_k_sweep(2048, 2048, 2048)
    print_block_k_sweep(block_k_sweep_rows)
    save_block_k_sweep_csv(block_k_sweep_rows)
    save_block_k_sweep_plot(block_k_sweep_rows)

    num_warps_sweep_rows = run_num_warps_sweep(2048, 2048, 2048)
    print_num_warps_sweep(num_warps_sweep_rows)
    save_num_warps_sweep_csv(num_warps_sweep_rows)
    save_num_warps_sweep_plot(num_warps_sweep_rows)
