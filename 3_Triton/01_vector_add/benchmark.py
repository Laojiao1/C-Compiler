from pathlib import Path

import torch
import triton
import triton.testing

if __package__:
    from .kernel import add
else:
    from kernel import add


DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE = torch.float32
WARMUP_MS = 100 # 预热 ms
REP_MS = 300 # 正式重复测量约 300 ms
QUANTILES = [0.2, 0.5, 0.8] # 返回 P20/P50/P80 latency
SUMMARY_SIZE = 2**24 # 汇总时测试的元素个数
BLOCK_SIZES = [256, 512, 1024, 2048]
RESULT_DIR = Path(__file__).with_name("benchmark_results")

# 有效带宽计算
def effective_gbps(size: int, element_size: int, latency_ms: float) -> float:
    """按 2 次读取 + 1 次写入计算算子语义上的有效带宽。"""
    logical_bytes = 3 * size * element_size
    return logical_bytes / latency_ms * 1e-6

# 计时操作
def measure(size: int, provider: str, block_size: int = 1024):
    """返回 P20/P50/P80 latency；计时函数内部包含输出张量分配。"""
    x = torch.rand(size, device=DEVICE, dtype=DTYPE)
    y = torch.rand(size, device=DEVICE, dtype=DTYPE)

    if provider == "torch":
        fn = lambda: x + y
    elif provider == "triton":
        fn = lambda: add(x, y, BLOCK_SIZE=block_size)
    else:
        raise ValueError(f"未知 provider: {provider}")

    p20_ms, p50_ms, p80_ms = triton.testing.do_bench(
        fn,
        warmup=WARMUP_MS,
        rep=REP_MS,
        quantiles=QUANTILES,
    )
    return p20_ms, p50_ms, p80_ms

# Triton 与 PyTorch 的带宽曲线
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        # 最大 2^26：三个 FP32 张量约占 768 MiB，适合当前 8 GiB Laptop GPU。
        x_vals=[2**i for i in range(12, 27)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        xlabel="Number of elements",
        ylabel="Effective GB/s",
        plot_name="vector-add-effective-bandwidth",
        args={},
    )
)
def bandwidth_report(size, provider):
    p20_ms, p50_ms, p80_ms = measure(size, provider)
    element_size = torch.empty((), dtype=DTYPE).element_size()

    median = effective_gbps(size, element_size, p50_ms)
    lower = effective_gbps(size, element_size, p80_ms)
    upper = effective_gbps(size, element_size, p20_ms)
    return median, lower, upper


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 27)],
        x_log=True,
        line_arg="block_size",
        line_vals=BLOCK_SIZES,
        line_names=[f"BLOCK_SIZE={value}" for value in BLOCK_SIZES],
        styles=[("red", "-"), ("orange", "-"), ("blue", "-"), ("purple", "-")],
        xlabel="Number of elements",
        ylabel="Effective GB/s",
        plot_name="vector-add-block-size-sweep",
        args={},
    )
)
def block_size_report(size, block_size):
    p20_ms, p50_ms, p80_ms = measure(size, "triton", block_size=block_size)
    element_size = torch.empty((), dtype=DTYPE).element_size()

    median = effective_gbps(size, element_size, p50_ms)
    lower = effective_gbps(size, element_size, p80_ms)
    upper = effective_gbps(size, element_size, p20_ms)
    return median, lower, upper


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


def print_summary(size: int = SUMMARY_SIZE):
    """在一个进入稳定带宽区的大 shape 上同时打印 latency、带宽和 speedup。"""
    element_size = torch.empty((), dtype=DTYPE).element_size()
    results = {}

    for provider in ("torch", "triton"):
        p20_ms, p50_ms, p80_ms = measure(size, provider)
        results[provider] = {
            "p20_ms": p20_ms,
            "p50_ms": p50_ms,
            "p80_ms": p80_ms,
            "gbps": effective_gbps(size, element_size, p50_ms),
        }

    speedup = results["torch"]["p50_ms"] / results["triton"]["p50_ms"]

    print(f"\n=== Summary at N={size:,} ===")
    print(f"{'Provider':<10} {'P20 (ms)':>10} {'P50 (ms)':>10} {'P80 (ms)':>10} {'Eff. GB/s':>12}")
    for provider, display_name in (("triton", "Triton"), ("torch", "PyTorch")):
        row = results[provider]
        print(
            f"{display_name:<10} "
            f"{row['p20_ms']:>10.4f} {row['p50_ms']:>10.4f} "
            f"{row['p80_ms']:>10.4f} {row['gbps']:>12.2f}"
        )
    print(f"Triton speedup over PyTorch (P50): {speedup:.3f}x")
    print("注意：这是包含输出张量分配的算子级延迟，不是 profiler 的纯 kernel time。")


if __name__ == "__main__":
    torch.manual_seed(0)
    print_environment()
    print_summary()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    bandwidth_report.run(
        print_data=True,
        show_plots=False,
        save_path=str(RESULT_DIR),
    )
    block_size_report.run(
        print_data=True,
        show_plots=False,
        save_path=str(RESULT_DIR),
    )
    print(f"\n曲线和 CSV 已保存到：{RESULT_DIR}")
