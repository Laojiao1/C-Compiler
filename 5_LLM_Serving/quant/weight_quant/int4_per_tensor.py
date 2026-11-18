"""
整个权重共用一个 scale"""
from pathlib import Path
import torch

# 反量化
def quantize_dequantize_int4(w: torch.Tensor) -> torch.Tensor:
    qmax = 7
    max_abs = w.abs().max()

    # 整个权重矩阵共用一个scale, 如果全零的话用scale=1，避免除零
    scale = torch.where(max_abs > 0, max_abs / qmax, torch.ones_like(max_abs))
    q = (w / scale).round().clamp(-qmax, qmax)
    w_hat = q * scale

    print(f"weight max_abs = {max_abs.item():.6e}")
    print(f"scale = {scale.item():.6e}")
    print(f"quantized range = [{int(q.min().item())}, {int(q.max().item())}]")
    print(f"zero ratio = {(q == 0).float().mean().item():.2%}")

    return w_hat

def relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return ((actual - reference).norm() / reference.norm()).item()

def main() -> None:
    sample_path = Path(__file__).with_name("down_proj_sample.pt")
    sample = torch.load(sample_path, map_location="cpu", weights_only=True)

    w = sample["weight"]
    bias = sample["bias"]
    x = sample["validation"]["x"]
    y_ref = sample["validation"]["y_ref"]

    print(f"W={tuple(w.shape)}, X={tuple(x.shape)}, Y_ref={tuple(y_ref.shape)}")

    # 先用原始权重重算一遍，检查数据和误差计算。
    y_baseline = x @ w.T
    if bias is not None:
        y_baseline += bias
    print(f"FP32 baseline relerr = {relative_error(y_baseline, y_ref):.6e}")

    # 固定验证输入，只替换量化再反量化后的权重。
    w_hat = quantize_dequantize_int4(w)
    y_hat = x @ w_hat.T
    if bias is not None:
        y_hat += bias
    print(f"INT4 per-tensor relerr = {relative_error(y_hat, y_ref):.6e}")


if __name__ == "__main__":
    main()