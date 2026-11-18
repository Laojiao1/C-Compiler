from pathlib import Path

import torch

QMAX = 7

def quantize_weight(w: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    n, k = w.shape
    if k % group_size != 0:
        raise ValueError(f"K={k} 必须能被 group_size={group_size} 整除")
    
    # 每个输出通道沿 K 维分组，每组 128 个权重共用一个 INT4 步长
    groups = w.reshape(n, k // group_size, group_size)
    max_abs = groups.abs().amax(dim=-1, keepdim=True)
    s = torch.where(max_abs > 0,  max_abs / QMAX , torch.ones_like(max_abs))
    q = (groups / s).round().clamp(-QMAX, QMAX) # 量化
    return (q * s).reshape_as(w) # 反量化


def channel_scale(act_mean: torch.Tensor, alpha: float) -> torch.Tensor:
    # 按激活幅值计算每个通道的保护力度（激活越大的通道，分配的 s 越大）
    scale = act_mean.clamp_min(1e-8).pow(alpha)
    # 用同一个常数归一化，保留通道间的相对比例。
    return scale / (scale.min() * scale.max()).sqrt()


def output_relerr(
    split: dict,
    w_hat: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> float:
    # 注意 x 要除以 s，因为 w_hat 是 W*s 量化再反量化后的权重
    y_hat = (split["x"] / scale) @ w_hat.T
    if bias is not None:
        y_hat += bias

    y_ref = split["y_ref"]
    return ((y_hat - y_ref).norm() / y_ref.norm()).item()


def main() -> None:
    path = Path(__file__).with_name("down_proj_sample.pt")
    sample = torch.load(path, map_location="cpu", weights_only=True)

    w = sample["weight"]
    bias = sample["bias"]
    calibration = sample["calibration"]
    validation = sample["validation"]

    # 找到激活值较大的通道（即识别那 1% 左右的离群值/显著通道）
    act_mean = calibration["x"].abs().mean(dim=0)

    calibration_errors = {}
    for alpha in (0.0, 0.25, 0.5, 0.75):
        # 按通道找出离群值的缩放 scale
        scale = channel_scale(act_mean, alpha)
        # 放大对应权重的通道，保护它们不被量化误差破坏
        w_hat = quantize_weight(w * scale)
        # 误差计算
        error = output_relerr(calibration, w_hat, scale, bias)
        calibration_errors[alpha] = error
        print(f"alpha={alpha:.2f}, calibration relerr={error:.4%}")

    # 先锁定校准集选中的 alpha，再比较固定验证集上的两条路径。
    best_alpha = min(calibration_errors, key=calibration_errors.__getitem__)
    baseline_scale = torch.ones_like(act_mean)
    awq_scale = channel_scale(act_mean, best_alpha)
    baseline_error = output_relerr(validation, quantize_weight(w * 1), baseline_scale, bias)
    awq_error = output_relerr(validation, quantize_weight(w * awq_scale), awq_scale, bias)

    print(f"selected alpha={best_alpha:.2f}")
    print(f"RTN per-group validation relerr={baseline_error:.4%}")
    print(f"AWQ scaling validation relerr={awq_error:.4%}")


if __name__ == "__main__":
    main()