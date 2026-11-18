"""
固定激活和权重量化粒度，比较直接 W8A8 与 SmoothQuant 后的 W8A8。由校准激活和权重的输入通道最大绝对值构造 scale `s`，先用 `alpha=0.5`。在量化前验证 `X @ W.T` 与 `(X / s) @ (W * s).T` 的 FP32 输出在浮点误差范围内一致，然后分别对缩放后的激活和权重做 INT8 量化再反量化。"""
from pathlib import Path

import torch


def relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return ((actual - reference).norm() / reference.norm()).item()


def fake_quant_int8(tensor: torch.Tensor, max_abs: torch.Tensor) -> torch.Tensor:
    qmax = 127
    scale = torch.where(max_abs > 0, max_abs / qmax, torch.ones_like(max_abs))
    q = (tensor / scale).round().clamp(-qmax, qmax)
    return q * scale

def main() -> None:
    sample_path = Path(__file__).with_name("down_proj_sample.pt")
    sample = torch.load(sample_path, map_location="cpu", weights_only=True)

    w = sample["weight"]                 # [N, K]
    x_cal = sample["calibration"]["x"]   # [T_cal, K]
    x_val = sample["validation"]["x"]    # [T_val, K]
    y_ref = sample["validation"]["y_ref"]
    bias = sample["bias"]

    # 激活跨 token 取最大值，权重跨输出通道取最大值。
    # 两边都保留输入通道 K，为每个输入通道计算一个 s。
    a = x_cal.abs().amax(dim=0)
    b = w.abs().amax(dim=0)
    assert a.shape == b.shape == (w.shape[1],)

    # alpha = 0.5 时，s = sqrt(a / b)。下限用于处理全零通道。
    s = (a.clamp_min(1e-8) / b.clamp_min(1e-8)).sqrt()

    # [K] 沿两个矩阵的最后一维广播。
    x_smooth = x_val / s
    w_smooth = w * s

    y_original = x_val @ w.T
    y_smooth = x_smooth @ w_smooth.T
    if bias is not None:
        y_original += bias
        y_smooth += bias

    print(f"s shape = {tuple(s.shape)}")
    print(f"s min / max = {s.min().item():.6e} / {s.max().item():.6e}")
    print(f"original vs saved relerr = {relative_error(y_original, y_ref):.6e}")
    print(f"smoothed vs original relerr = {relative_error(y_smooth, y_original):.6e}")

    x_cal_smooth = x_cal / s

    for name, x_cal_run, x_val_run, w_run in (
        ("Direct W8A8", x_cal, x_val, w),
        ("SmoothQuant W8A8", x_cal_smooth, x_smooth, w_smooth),
    ):
        # 激活共用一个阈值，从校准集取得。
        x_max = x_cal_run.abs().amax()

        # W 是 [N, K]。每个输出通道独立定标，得到 [N, 1]。
        w_max = w_run.abs().amax(dim=1, keepdim=True)

        x_hat = fake_quant_int8(x_val_run, x_max)
        w_hat = fake_quant_int8(w_run, w_max)
        y_hat = x_hat @ w_hat.T
        if bias is not None:
            y_hat += bias

        print(
            f"{name}: "
            f"calibration X max={x_max.item():.6e}, "
            f"W absmax={w_max.max().item():.6e}, "
            f"validation relerr={relative_error(y_hat, y_original):.6e}"
        )


if __name__ == "__main__":

    main()