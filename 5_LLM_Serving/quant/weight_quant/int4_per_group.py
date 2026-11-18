"""
按输出行、沿输入通道每 128 个权重分组，并添加一组 clipping 对照
"""
from pathlib import Path

import torch


def quantize_dequantize_int4_per_group(
    w: torch.Tensor, group_size: int = 128, clip_ratio: float = 1.0
) -> torch.Tensor:
    qmax = 7
    n, k = w.shape
    if k % group_size != 0:
        raise ValueError(f"K={k} 必须能被 group_size={group_size} 整除")

    # 每个输出通道独立分组，每组包含连续的 128 个输入通道权重。
    groups = w.reshape(n, k // group_size, group_size)

    # 沿组内维度求最大值。保留最后一维，便于把 scale 广播给组内权重。
    max_abs = groups.abs().amax(dim=-1, keepdim=True)
    # 尝试 clip：判断小权重舍入的更准，能否抵消大权重被截断更多的误差？
    threshold = max_abs * clip_ratio
    scale = torch.where(threshold > 0, threshold / qmax, torch.ones_like(threshold))

    q = (groups / scale).round().clamp(-qmax, qmax)
    w_hat = (q * scale).reshape_as(w)

    print(f"grouped W = {tuple(groups.shape)}, scale = {tuple(scale.shape)}")
    print(f"scale median = {scale.median().item():.6e}")
    print(f"scale max = {scale.max().item():.6e}")
    print(f"zero ratio = {(q == 0).float().mean().item():.2%}")
    return w_hat


def main() -> None:
    sample_path = Path(__file__).with_name("down_proj_sample.pt")
    sample = torch.load(sample_path, map_location="cpu", weights_only=True)

    w = sample["weight"]
    x = sample["validation"]["x"]
    y_ref = sample["validation"]["y_ref"]
    bias = sample["bias"]

    w_hat = quantize_dequantize_int4_per_group(w)
    y_hat = x @ w_hat.T
    if bias is not None:
        y_hat += bias

    relerr = (y_hat - y_ref).norm() / y_ref.norm()
    print(f"INT4 per-group relerr = {relerr.item():.6e}")

    def output_relerr(w_hat: torch.Tensor, split: str) -> float:
        x = sample[split]["x"]
        y_ref = sample[split]["y_ref"]
        y_hat = x @ w_hat.T
        if bias is not None:
            y_hat += bias
        return ((y_hat - y_ref).norm() / y_ref.norm()).item()

    calibration_errors = {}
    for ratio in (1.0, 0.95, 0.9, 0.8):
        w_hat = quantize_dequantize_int4_per_group(w, clip_ratio=ratio)
        error = output_relerr(w_hat, "calibration")
        calibration_errors[ratio] = error
        print(f"ratio={ratio:.2f}, calibration relerr={error:.6e}")

    best_ratio = min(calibration_errors, key=calibration_errors.__getitem__)
    w_hat = quantize_dequantize_int4_per_group(w, clip_ratio=best_ratio)
    print(f"selected ratio={best_ratio:.2f}")
    print(f"validation relerr={output_relerr(w_hat, 'validation'):.6e}")


if __name__ == "__main__":
    main()