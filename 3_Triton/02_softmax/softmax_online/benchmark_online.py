import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import triton
import triton.testing

if __package__:
    from .kernel_online import softmax_online
else:
    from kernel_online import softmax_online


DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE = torch.float32
BLOCK_SIZE = 1024
WARMUP_MS = 100
REP_MS = 300
QUANTILES = [0.2, 0.5, 0.8]
RESULT_DIR = Path(__file__).with_name("benchmark_results")


def semantic_gbps(M: int, N: int, latency_ms: float) -> float:
    """统一按读取输入一次、写入输出一次计算，便于直接比较完成 Softmax 的速度。"""
    logical_bytes = 2 * M * N * torch.empty((), dtype=DTYPE).element_size()
    return logical_bytes / latency_ms * 1e-6


def online_algorithmic_gbps(M: int, N: int, latency_ms: float) -> float:
    """Online 实际执行两次输入读取和一次输出写入，对应 3 份张量流量。"""
    algorithmic_bytes = 3 * M * N * torch.empty((), dtype=DTYPE).element_size()
    return algorithmic_bytes / latency_ms * 1e-6


def measure(fn):
    """返回 P20/P50/P80 算子级延迟，单位为毫秒。"""
    return triton.testing.do_bench(
        fn,
        warmup=WARMUP_MS,
        rep=REP_MS,
        quantiles=QUANTILES,
    )


def benchmark_case(M: int, N: int) -> dict:
    x = torch.randn((M, N), device=DEVICE, dtype=DTYPE)

    # correctness 不放进计时区间；每个 shape 在 benchmark 前先验证一次。
    expected = F.softmax(x, dim=1)
    actual = softmax_online(x, BLOCK_SIZE=BLOCK_SIZE)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    del expected, actual

    torch_p20, torch_p50, torch_p80 = measure(lambda: F.softmax(x, dim=1))
    online_p20, online_p50, online_p80 = measure(
        lambda: softmax_online(x, BLOCK_SIZE=BLOCK_SIZE)
    )

    return {
        "M": M,
        "N": N,
        "chunks_per_row": triton.cdiv(N, BLOCK_SIZE),
        "torch_p20_ms": torch_p20,
        "torch_p50_ms": torch_p50,
        "torch_p80_ms": torch_p80,
        "online_p20_ms": online_p20,
        "online_p50_ms": online_p50,
        "online_p80_ms": online_p80,
        "online_speedup": torch_p50 / online_p50,
        "torch_semantic_gbps": semantic_gbps(M, N, torch_p50),
        "online_semantic_gbps": semantic_gbps(M, N, online_p50),
        "online_algorithmic_gbps": online_algorithmic_gbps(M, N, online_p50),
    }


def print_environment():
    props = torch.cuda.get_device_properties(DEVICE)
    print("\n=== Environment ===")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Triton:       {triton.__version__}")
    print(f"CUDA:         {torch.version.cuda}")
    print(f"GPU:          {props.name}")
    print(f"SM count:     {props.multi_processor_count}")
    print(f"dtype:        {DTYPE}")
    print(f"BLOCK_SIZE:   {BLOCK_SIZE}")
    print(f"warmup/rep:   {WARMUP_MS}/{REP_MS} ms")


def print_results(title: str, rows: list[dict]):
    print(f"\n=== {title} ===")
    print(
        f"{'M':>6} {'N':>9} {'Chunks':>7} "
        f"{'Torch ms':>10} {'Online ms':>10} {'Speedup':>9} "
        f"{'Torch Sem GB/s':>15} {'Online Sem GB/s':>16} {'Online Alg GB/s':>16}"
    )
    for row in rows:
        print(
            f"{row['M']:>6} {row['N']:>9} {row['chunks_per_row']:>7} "
            f"{row['torch_p50_ms']:>10.4f} {row['online_p50_ms']:>10.4f} "
            f"{row['online_speedup']:>9.3f} "
            f"{row['torch_semantic_gbps']:>15.2f} "
            f"{row['online_semantic_gbps']:>16.2f} "
            f"{row['online_algorithmic_gbps']:>16.2f}"
        )
    print("Speedup > 1 表示 Online 更快，< 1 表示 PyTorch 更快。")


def save_csv(filename: str, rows: list[dict]):
    path = RESULT_DIR / filename
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"结果已保存：{path}")


def save_plot(filename: str, title: str, rows: list[dict]):
    """左图比较延迟，右图用 1.0x 基准线展示 Online 是否更快。"""
    positions = list(range(len(rows)))
    labels = [f"{row['M']}x{row['N']}" for row in rows]
    torch_latency = [row["torch_p50_ms"] for row in rows]
    online_latency = [row["online_p50_ms"] for row in rows]
    speedups = [row["online_speedup"] for row in rows]

    figure, (latency_axis, speedup_axis) = plt.subplots(1, 2, figsize=(14, 5))

    latency_axis.plot(positions, torch_latency, marker="o", label="PyTorch F.softmax")
    latency_axis.plot(positions, online_latency, marker="o", label="Triton Online")
    latency_axis.set_yscale("log")
    latency_axis.set_ylabel("P50 latency (ms, log scale)")
    latency_axis.set_title("Latency: lower is better")
    latency_axis.grid(True, alpha=0.3)
    latency_axis.legend()

    speedup_axis.plot(positions, speedups, marker="o", color="purple")
    speedup_axis.axhline(1.0, color="black", linestyle="--", label="1.0x")
    speedup_axis.set_ylabel("Online speedup over PyTorch")
    speedup_axis.set_title("Speedup: above 1.0x is better")
    speedup_axis.grid(True, alpha=0.3)
    speedup_axis.legend()

    for axis in (latency_axis, speedup_axis):
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.set_xlabel("Matrix shape (M x N)")

    figure.suptitle(title)
    figure.tight_layout()
    path = RESULT_DIR / filename
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"图像已保存：{path}")


def run_fixed_rows_experiment():
    """固定 M，只增加行长 N，观察每行 chunk 数增加的代价。"""
    M = 256
    shapes = [(M, N) for N in (4096, 8192, 16384, 32768, 65536, 131072)]
    rows = [benchmark_case(M, N) for M, N in shapes]
    print_results("Experiment 1: Fixed M=256, increase N", rows)
    save_csv("online-softmax-fixed-rows.csv", rows)
    save_plot(
        "online-softmax-fixed-rows.png",
        "Online Softmax: fixed M=256, increase row length",
        rows,
    )


def run_fixed_elements_experiment():
    """固定 M*N，减少行数并增加行长，观察 program 并行度变化。"""
    shapes = [
        (4096, 4096),
        (2048, 8192),
        (1024, 16384),
        (512, 32768),
        (256, 65536),
        (128, 131072),
        (64, 262144),
        (32, 524288),
        (16, 1048576),
    ]
    rows = [benchmark_case(M, N) for M, N in shapes]
    print_results("Experiment 2: Fixed M*N=2^24, change matrix shape", rows)
    save_csv("online-softmax-fixed-elements.csv", rows)
    save_plot(
        "online-softmax-fixed-elements.png",
        "Online Softmax: fixed total elements, change matrix shape",
        rows,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print_environment()
    run_fixed_rows_experiment()
    run_fixed_elements_experiment()
