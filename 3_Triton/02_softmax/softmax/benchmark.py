from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.testing
from kernel import softmax

# 配置参数
DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE = torch.float32
WARMUP_MS = 100
REP_MS = 300
QUANTILES = [0.2, 0.5, 0.8]
SUMMARY_M = 4096
SUMMARY_N = 4096
RESULT_DIR = Path(__file__).with_name("benchmark_results")

# 朴素的 Softmax 实现，使用多个 PyTorch primitive 实现数值稳定 Softmax，作为未融合基线。
def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """用多个 PyTorch primitive 实现数值稳定 Softmax，作为未融合基线。"""
    row_max = torch.max(x, dim=1, keepdim=True).values
    numerator = torch.exp(x - row_max)
    denominator = torch.sum(numerator, dim=1, keepdim=True)
    return numerator / denominator

# 计算有效带宽，单位为 GB/s。
def effective_gbps(M: int, N: int, element_size: int, latency_ms: float) -> float:
    """按 Softmax 的逻辑输入一次、输出一次计算有效带宽。"""
    logical_bytes = 2 * M * N * element_size
    return logical_bytes / latency_ms * 1e-6

# 计算算子级 latency，单位为毫秒。
def measure(M: int, N: int, provider: str):
    """返回 P20/P50/P80 算子级 latency，单位为毫秒。"""
    x = torch.randn((M, N), device=DEVICE, dtype=DTYPE)

    if provider == "torch":
        fn = lambda: F.softmax(x, dim=1)
    elif provider == "naive":
        fn = lambda: naive_softmax(x)
    elif provider == "triton":
        fn = lambda: softmax(x)
    else:
        raise ValueError(f"未知 provider: {provider}")

    # do_bench 反复调用 fn，返回 P20/P50/P80 latency，单位为毫秒。
    p20_ms, p50_ms, p80_ms = triton.testing.do_bench(
        fn,
        warmup=WARMUP_MS,
        rep=REP_MS,
        quantiles=QUANTILES,
    )
    return p20_ms, p50_ms, p80_ms

# Triton 测试框架的性能报告装饰器，生成带宽曲线。
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        # 线性扫描会跨过多个 next_power_of_2 边界，便于观察 padding 代价。
        x_vals=[128 * i for i in range(1, 65)],
        line_arg="provider",
        line_vals=["triton", "torch", "naive"],
        line_names=["Triton Fused", "PyTorch F.softmax", "PyTorch Naive"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        xlabel="Number of columns (N)",
        ylabel="Effective GB/s",
        plot_name="softmax-effective-bandwidth",
        args={"M": 4096},
    )
)
# 计算有效带宽，单位为 GB/s。
def bandwidth_report(M, N, provider):
    p20_ms, p50_ms, p80_ms = measure(M, N, provider)
    element_size = torch.empty((), dtype=DTYPE).element_size()

    median = effective_gbps(M, N, element_size, p50_ms)
    lower = effective_gbps(M, N, element_size, p80_ms)
    upper = effective_gbps(M, N, element_size, p20_ms)
    return median, lower, upper

# 打印环境信息，包括 PyTorch、Triton 版本，GPU 型号、计算能力、SM 数量、显存大小等
def print_environment():
    props = torch.cuda.get_device_properties(DEVICE)
    print("\n=== Environment ===")
    print(f"PyTorch:           {torch.__version__}")
    print(f"Triton:            {triton.__version__}")
    print(f"PyTorch CUDA:      {torch.version.cuda}")
    print(f"GPU:               {props.name}")
    print(f"Compute capability:{props.major}.{props.minor}")
    print(f"SM count:          {props.multi_processor_count}")
    print(f"VRAM:              {props.total_memory / 2**30:.2f} GiB")
    print(f"dtype:             {DTYPE}")
    print(f"warmup/rep:        {WARMUP_MS}/{REP_MS} ms")

# 打印算子级 latency 和有效带宽的汇总表
def print_summary(M: int = SUMMARY_M, N: int = SUMMARY_N):
    element_size = torch.empty((), dtype=DTYPE).element_size()
    results = {}

    for provider in ("triton", "torch", "naive"):
        p20_ms, p50_ms, p80_ms = measure(M, N, provider)
        results[provider] = {
            "p20_ms": p20_ms,
            "p50_ms": p50_ms,
            "p80_ms": p80_ms,
            "gbps": effective_gbps(M, N, element_size, p50_ms),
        }

    torch_speedup = results["torch"]["p50_ms"] / results["triton"]["p50_ms"]
    fusion_speedup = results["naive"]["p50_ms"] / results["triton"]["p50_ms"]

    print(f"\n=== Summary at M={M:,}, N={N:,} ===")
    print(f"{'Provider':<18} {'P20 (ms)':>10} {'P50 (ms)':>10} {'P80 (ms)':>10} {'Eff. GB/s':>12}")
    names = {
        "triton": "Triton Fused",
        "torch": "PyTorch F.softmax",
        "naive": "PyTorch Naive",
    }
    for provider in ("triton", "torch", "naive"):
        row = results[provider]
        print(
            f"{names[provider]:<18} "
            f"{row['p20_ms']:>10.4f} {row['p50_ms']:>10.4f} "
            f"{row['p80_ms']:>10.4f} {row['gbps']:>12.2f}"
        )

    print(f"Triton speedup over PyTorch F.softmax (P50): {torch_speedup:.3f}x")
    print(f"Triton speedup over PyTorch Naive (P50):     {fusion_speedup:.3f}x")
    print("注意：三条曲线统一使用输入+输出的逻辑字节量；Naive 的真实中间流量更大。")


if __name__ == "__main__":
    torch.manual_seed(0)
    print_environment()
    print_summary()

    # 生成带宽曲线和 CSV 文件
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    bandwidth_report.run(
        print_data=True,
        show_plots=False,
        save_path=str(RESULT_DIR),
    )
    print(f"\n曲线和 CSV 已保存到：{RESULT_DIR}")
