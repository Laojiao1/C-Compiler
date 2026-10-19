import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import triton
import triton.testing

if __package__:
    from .kernel import layer_norm, layer_norm_autotuned
else:
    # 兼容直接执行：python benchmark.py
    from kernel import layer_norm, layer_norm_autotuned


DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE = torch.float16
EPS = 1e-5

WARMUP_MS = 100
REP_MS = 300
QUANTILES = [0.2, 0.5, 0.8]
RESULT_DIR = Path(__file__).with_name("benchmark_results")

# 固定行数，专门观察跨越 2 的幂后 padding 和 reduction 宽度的变化。
BOUNDARY_M = 4096
BOUNDARY_N_VALUES = [
    127,
    128,
    129,
    511,
    512,
    513,
    1023,
    1024,
    1025,
    2047,
    2048,
    2049,
]

# 固定总元素数，只改变行数与特征维度，避免把总工作量变化误判为 shape 效应。
FIXED_NUMEL = 2**22
FIXED_NUMEL_N_VALUES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

# 固定总元素数，在短、中、长三种特征维度上只改变 num_warps。
WARP_SWEEP_SHAPES = [
    (8192, 512),
    (2048, 2048),
    (512, 8192),
    (128, 32768),
]
NUM_WARPS_VALUES = [1, 2, 4, 8]


def core_effective_gbps(
    m: int,
    n: int,
    element_size: int,
    latency_ms: float,
) -> float:
    """按读取主输入 x 一次、写回输出 y 一次计算统一比较口径。"""
    core_bytes = 2 * m * n * element_size
    return core_bytes / latency_ms * 1e-6


def logical_effective_gbps(
    m: int,
    n: int,
    element_size: int,
    latency_ms: float,
) -> float:
    """计入 x/weight/bias/y 的源码级逻辑流量，不代表真实 DRAM 流量。"""
    logical_bytes = 4 * m * n * element_size
    return logical_bytes / latency_ms * 1e-6


def measure(fn) -> tuple[float, float, float]:
    """返回 P20、P50、P80 算子级延迟，单位为毫秒。"""
    return triton.testing.do_bench(
        fn,
        warmup=WARMUP_MS,
        rep=REP_MS,
        quantiles=QUANTILES,
    )


def make_inputs(m: int, n: int):
    x = torch.randn((m, n), device=DEVICE, dtype=DTYPE)
    weight = torch.randn(n, device=DEVICE, dtype=DTYPE)
    bias = torch.randn(n, device=DEVICE, dtype=DTYPE)
    return x, weight, bias


def check_correctness(x, weight, bias, num_warps: int = 4):
    """正确性调用也会完成对应 BLOCK_SIZE/num_warps 的首次 JIT 编译。"""
    n_cols = x.shape[1]
    expected = F.layer_norm(x, (n_cols,), weight, bias, eps=EPS)
    actual = layer_norm(x, weight, bias, eps=EPS, num_warps=num_warps)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def check_autotuned_correctness(x, weight, bias):
    n_cols = x.shape[1]
    expected = F.layer_norm(x, (n_cols,), weight, bias, eps=EPS)
    actual = layer_norm_autotuned(x, weight, bias, eps=EPS)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def benchmark_torch_vs_triton(m: int, n: int, num_warps: int = 4) -> dict:
    x, weight, bias = make_inputs(m, n)
    check_correctness(x, weight, bias, num_warps=num_warps)
    # 这次调用会完成当前 M/N 下的四个候选配置测量，并缓存最佳配置。
    check_autotuned_correctness(x, weight, bias)

    torch_p20, torch_p50, torch_p80 = measure(
        lambda: F.layer_norm(x, (n,), weight, bias, eps=EPS)
    )
    manual_p20, manual_p50, manual_p80 = measure(
        lambda: layer_norm(x, weight, bias, eps=EPS, num_warps=num_warps)
    )
    autotuned_p20, autotuned_p50, autotuned_p80 = measure(
        lambda: layer_norm_autotuned(x, weight, bias, eps=EPS)
    )

    element_size = x.element_size()
    block_size = triton.next_power_of_2(n)
    return {
        "M": m,
        "N": n,
        "BLOCK_SIZE": block_size,
        "manual_num_warps": num_warps,
        "torch_p20_ms": torch_p20,
        "torch_p50_ms": torch_p50,
        "torch_p80_ms": torch_p80,
        "manual_p20_ms": manual_p20,
        "manual_p50_ms": manual_p50,
        "manual_p80_ms": manual_p80,
        "autotuned_p20_ms": autotuned_p20,
        "autotuned_p50_ms": autotuned_p50,
        "autotuned_p80_ms": autotuned_p80,
        "torch_core_gbps": core_effective_gbps(m, n, element_size, torch_p50),
        "manual_core_gbps": core_effective_gbps(m, n, element_size, manual_p50),
        "autotuned_core_gbps": core_effective_gbps(
            m,
            n,
            element_size,
            autotuned_p50,
        ),
        "autotuned_logical_gbps": logical_effective_gbps(
            m,
            n,
            element_size,
            autotuned_p50,
        ),
        "manual_speedup": torch_p50 / manual_p50,
        "autotuned_speedup": torch_p50 / autotuned_p50,
    }


def benchmark_warp_shape(m: int, n: int) -> list[dict]:
    """同一 shape 复用输入和 PyTorch baseline，只改变 Triton num_warps。"""
    x, weight, bias = make_inputs(m, n)

    # 先完成所有配置的正确性检查和 JIT，避免编译活动进入后续计时阶段。
    for num_warps in NUM_WARPS_VALUES:
        check_correctness(x, weight, bias, num_warps=num_warps)

    # 同一个 shape 只测一次 PyTorch，所有 warp 配置共享这个参考。
    torch_p20, torch_p50, torch_p80 = measure(
        lambda: F.layer_norm(x, (n,), weight, bias, eps=EPS)
    )

    element_size = x.element_size()
    rows = []
    for num_warps in NUM_WARPS_VALUES:
        triton_p20, triton_p50, triton_p80 = measure(
            lambda num_warps=num_warps: layer_norm(
                x,
                weight,
                bias,
                eps=EPS,
                num_warps=num_warps,
            )
        )
        rows.append(
            {
                "M": m,
                "N": n,
                "BLOCK_SIZE": triton.next_power_of_2(n),
                "num_warps": num_warps,
                "torch_p20_ms": torch_p20,
                "torch_p50_ms": torch_p50,
                "torch_p80_ms": torch_p80,
                "triton_p20_ms": triton_p20,
                "triton_p50_ms": triton_p50,
                "triton_p80_ms": triton_p80,
                "triton_core_gbps": core_effective_gbps(
                    m,
                    n,
                    element_size,
                    triton_p50,
                ),
                "speedup": torch_p50 / triton_p50,
            }
        )
    return rows


def print_environment():
    properties = torch.cuda.get_device_properties(DEVICE)
    print("\n=== Environment ===")
    print(f"PyTorch:            {torch.__version__}")
    print(f"Triton:             {triton.__version__}")
    print(f"CUDA:               {torch.version.cuda}")
    print(f"GPU:                {properties.name}")
    print(f"Compute capability: {properties.major}.{properties.minor}")
    print(f"SM count:           {properties.multi_processor_count}")
    print(f"VRAM:               {properties.total_memory / 2**30:.2f} GiB")
    print(f"dtype:              {DTYPE}")
    print(f"warmup/rep:         {WARMUP_MS}/{REP_MS} ms")
    print("计时范围：Python wrapper + 输出分配 + kernel launch + kernel 执行")


def print_comparison(title: str, rows: list[dict]):
    print(f"\n=== {title} ===")
    print(
        f"{'M':>7} {'N':>7} {'BLOCK':>7} "
        f"{'Torch P50':>10} {'Manual P50':>11} {'Auto P50':>10} "
        f"{'Torch GB/s':>11} {'Manual GB/s':>11} {'Auto GB/s':>10} "
        f"{'Manual x':>8} {'Auto x':>8}"
    )
    for row in rows:
        print(
            f"{row['M']:>7} {row['N']:>7} {row['BLOCK_SIZE']:>7} "
            f"{row['torch_p50_ms']:>10.4f} {row['manual_p50_ms']:>11.4f} "
            f"{row['autotuned_p50_ms']:>10.4f} "
            f"{row['torch_core_gbps']:>11.2f} {row['manual_core_gbps']:>11.2f} "
            f"{row['autotuned_core_gbps']:>10.2f} "
            f"{row['manual_speedup']:>8.3f} {row['autotuned_speedup']:>8.3f}"
        )
    print("Manual 固定 w4；Auto 在 w1/w2/w4/w8 中按 M/N 实测选择。")
    print("GB/s 使用 2MNs 核心张量口径；speedup > 1 表示 Triton 更快。")


def print_warp_sweep(rows: list[dict]):
    print("\n=== Experiment 3: num_warps sweep ===")
    print(
        f"{'M':>7} {'N':>7} {'BLOCK':>7} {'warps':>6} "
        f"{'Torch P20/P50/P80':>24} {'Triton P20/P50/P80':>25} "
        f"{'Triton GB/s':>12} {'Speedup':>8}"
    )
    for row in rows:
        print(
            f"{row['M']:>7} {row['N']:>7} {row['BLOCK_SIZE']:>7} "
            f"{row['num_warps']:>6} "
            f"{row['torch_p20_ms']:>7.4f}/{row['torch_p50_ms']:^7.4f}/{row['torch_p80_ms']:<7.4f} "
            f"{row['triton_p20_ms']:>7.4f}/{row['triton_p50_ms']:^7.4f}/{row['triton_p80_ms']:<7.4f} "
            f"{row['triton_core_gbps']:>12.2f} {row['speedup']:>8.3f}"
        )
    print("同一 shape 内只改变 num_warps；不要跨 shape 比较最佳 warp。")


def save_csv(rows: list[dict], filename: str):
    path = RESULT_DIR / filename
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV 已保存：{path}")


def save_comparison_plot(rows: list[dict], filename: str, title: str):
    """绘制三方 P50 latency 与核心 Effective GB/s。"""
    positions = list(range(len(rows)))
    labels = [str(row["N"]) for row in rows]

    figure, (latency_axis, bandwidth_axis) = plt.subplots(
        1,
        2,
        figsize=(16, 6),
    )

    providers = [
        ("PyTorch", "torch_p50_ms", "torch_core_gbps", "green"),
        ("Triton manual w4", "manual_p50_ms", "manual_core_gbps", "blue"),
        ("Triton autotuned", "autotuned_p50_ms", "autotuned_core_gbps", "red"),
    ]
    for name, latency_key, bandwidth_key, color in providers:
        latency_axis.plot(
            positions,
            [row[latency_key] for row in rows],
            marker="o",
            label=name,
            color=color,
        )
        bandwidth_axis.plot(
            positions,
            [row[bandwidth_key] for row in rows],
            marker="o",
            label=name,
            color=color,
        )

    latency_axis.set_yscale("log")
    latency_axis.set_ylabel("P50 latency (ms, log scale)")
    latency_axis.set_title("Latency: lower is better")
    bandwidth_axis.set_ylabel("Core effective GB/s (2MNs)")
    bandwidth_axis.set_title("Bandwidth: higher is better")

    for axis in (latency_axis, bandwidth_axis):
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.set_xlabel("Feature dimension N")
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle(title)
    figure.tight_layout()
    path = RESULT_DIR / filename
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"图像已保存：{path}")


def save_warp_sweep_plot(rows: list[dict]):
    """每条曲线对应一个 shape，只比较该 shape 内不同 num_warps。"""
    figure, axis = plt.subplots(figsize=(10, 6))

    for m, n in WARP_SWEEP_SHAPES:
        shape_rows = [
            row
            for row in rows
            if row["M"] == m and row["N"] == n
        ]
        axis.plot(
            [row["num_warps"] for row in shape_rows],
            [row["triton_p50_ms"] for row in shape_rows],
            marker="o",
            label=f"M={m}, N={n}",
        )

    axis.set_xticks(NUM_WARPS_VALUES)
    axis.set_xlabel("num_warps")
    axis.set_ylabel("Triton P50 latency (ms)")
    axis.set_title("LayerNorm num_warps sweep: lower is better")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    path = RESULT_DIR / "layernorm-warp-sweep.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"图像已保存：{path}")


@torch.inference_mode()
def main():
    torch.manual_seed(0)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print_environment()

    boundary_rows = [
        benchmark_torch_vs_triton(BOUNDARY_M, n)
        for n in BOUNDARY_N_VALUES
    ]
    print_comparison(
        "Experiment 1: fixed M, power-of-two boundaries",
        boundary_rows,
    )
    save_csv(boundary_rows, "layernorm-boundary.csv")
    save_comparison_plot(
        boundary_rows,
        "layernorm-boundary.png",
        "LayerNorm: power-of-two boundaries (fixed M=4096)",
    )

    fixed_numel_rows = [
        benchmark_torch_vs_triton(FIXED_NUMEL // n, n)
        for n in FIXED_NUMEL_N_VALUES
    ]
    print_comparison(
        f"Experiment 2: fixed numel={FIXED_NUMEL:,}",
        fixed_numel_rows,
    )
    save_csv(fixed_numel_rows, "layernorm-fixed-numel.csv")
    save_comparison_plot(
        fixed_numel_rows,
        "layernorm-fixed-numel.png",
        f"LayerNorm: fixed numel={FIXED_NUMEL:,}",
    )

    warp_rows = [
        row
        for m, n in WARP_SWEEP_SHAPES
        for row in benchmark_warp_shape(m, n)
    ]
    print_warp_sweep(warp_rows)
    save_csv(warp_rows, "layernorm-warp-sweep.csv")
    save_warp_sweep_plot(warp_rows)

    print("\n说明：autotuned logical GB/s 已计算但不作为主表指标，因为 affine 参数可能命中 L2。")
    print(f"CSV 和图片已保存到：{RESULT_DIR}")


if __name__ == "__main__":
    main()
