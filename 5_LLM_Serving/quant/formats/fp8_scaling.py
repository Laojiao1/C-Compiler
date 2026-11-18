"""
FP8 通过指数来决定每个数量化时的附近刻度，越小的数，越精细，越大的数，勉强兜住就行，主要是为了保护小权重。
而INT8相当于全程用一把均衡的刻度，为了兜住离群值，步长被迫拉大，导致更多的小权重被一刀切到了0，大量微小特征值的信息丢失，相对误差要比FP8高.
"""

from pathlib import Path

import torch


FP8 = torch.float8_e4m3fn # (-1)^S + 2^(E-7) * (1 + M/8)

def quantize_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 约定 scale 表示反量化乘数，INT8 在这个单位下使用等距网格。
    amax = w.abs().amax()
    scale = torch.where(amax > 0, amax / 127, torch.ones_like(amax))
    codes = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return codes, scale


def quantize_e4m3(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 先把张量范围映射到 FP8 的有限范围，再由 dtype 转换执行浮点舍入。
    # 和 INT8 一样，每个张量只用一个外部 scale。
    limit = torch.finfo(FP8).max # 448
    amax = w.abs().amax()
    scale = torch.where(amax > 0, amax / limit, torch.ones_like(amax))
    codes = (w / scale).clamp(-limit, limit).to(FP8)
    return codes, scale


def show_small_example() -> None:
    # 最大值取 448，因此这一组 E4M3 的 scale 恰好为 1，方便读位模式。
    w = torch.tensor([1.0, 1.0625, 1.125, 1.1875, 64.0, 65.0, 64.0])
    int8, int8_scale = quantize_int8(w)
    fp8, fp8_scale = quantize_e4m3(w)
    print(f"toy scales: INT8={int8_scale.item():.6g}, E4M3={fp8_scale.item():.6g}")
    print(f"{'original':>10s} {'INT8 restored':>15s} {'E4M3 restored':>15s} {'E4M3 bits':>10s}")
    # 用 view 看实际存储的字节，和 FP8 转回 FP32 得到的数值分别对照。
    for value, fixed, floating, bits in zip(
        w.tolist(),
        (int8.float() * int8_scale).tolist(),
        (fp8.float() * fp8_scale).tolist(),
        fp8.view(torch.uint8).tolist(),
    ):
        raw = f"{bits:08b}"
        fields = f"{raw[0]}_{raw[1:5]}_{raw[5:]}"
        print(f"{value:10g} {fixed:15.6g} {floating:15.6g} {fields:>10s}")


def show_layer_error() -> None:
    path = Path(__file__).parents[1] / "weight_quant" / "down_proj_sample.pt"
    sample = torch.load(path, map_location="cpu", weights_only=True)
    w, bias = sample["weight"], sample["bias"]
    x = sample["validation"]["x"]
    ref = sample["validation"]["y_ref"]
    print(f"down_proj: W={tuple(w.shape)}, X={tuple(x.shape)}, weights-only")

    for name, quantize in (("INT8", quantize_int8), ("E4M3", quantize_e4m3)):
        codes, scale = quantize(w)
        restored = codes.float() * scale
        # 固定原始激活。将反量化权重交给相同的 FP32 GEMM，只比较格式造成的误差。
        actual = x @ restored.T
        if bias is not None:
            actual += bias
        zeros = (restored == 0).float().mean().item()
        relerr = ((actual - ref).norm() / ref.norm()).item()
        print(f"{name}: scale={scale.item():.6e}, zeros={zeros:.4%}, output relerr={relerr:.4%}")


if __name__ == "__main__":
    show_small_example()
    show_layer_error()

