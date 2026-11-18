from pathlib import Path

import torch

from .fp8_scaling import quantize_e4m3

# 实现就近偶数舍入（银行家舍入算法）
def round_e2m1(x: torch.Tensor) -> torch.Tensor:
    levels = x.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]) # FP4-E2M1 所能表示的数
    midpoints = (levels[:-1] + levels[1:]) * 0.5 # 算出相邻两个合法刻度之间的 7 个分界中点：[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    magnitude = x.abs().clamp_max(6.0)

    # bucketize 在恰好等于中点时选低一档。若低档编码为奇数，往上走一档，就实现 round-to-nearest-even
    index = torch.bucketize(magnitude, midpoints)
    boundary = midpoints[index.clamp_max(midpoints.numel() - 1)]
    index += ((index % 2 == 1) & (magnitude == boundary)).long() # 奇数进位，偶数保留
    return torch.copysign(levels[index], x) # 查表还原符号


# 每 16 个元素共享一个 E4M3 scale，再乘整个张量的 FP32 scale
def nvfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    blocks = w.reshape(w.shape[0], -1, 16) # 沿着最后一个维度，每16个元素切成一组
    block_max = blocks.abs().amax(dim=-1, keepdim=True) # 找出每组 16 个权重里的最大绝对值 block_max
    fp8 = torch.float8_e4m3fn
    limit = torch.finfo(fp8).max

    global_scale = block_max.amax() / (6 * limit) # 让最大的一块 global_scale 刚好等于 limit：global_max/6 = global_scale * limit
    global_scale = torch.where(global_scale > 0, global_scale, 1.0)
    block_scale = (block_max / (6 * global_scale)).clamp_max(limit).to(fp8)
    total_scale = global_scale * block_scale.float() # 总缩放

    # E4M3 块 scale 若舍入成零，避免除零，反量化仍乘回实际的零 scale。
    safe_scale = torch.where(total_scale > 0, total_scale, 1.0)
    restored = round_e2m1(blocks / safe_scale) * total_scale

    return restored.reshape_as(w), block_scale.float(), global_scale


# 每 32 个元素共享一个 E8M0 scale，scale 只能取 2 的整数次幂
def mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = w.reshape(w.shape[0], -1, 32)
    block_max = blocks.abs().amax(dim=-1, keepdim=True)

    # frexp 返回 floor(log2(amax)) + 1，E2M1 最大指数为 2, 因此这里减 3
    _, exponent = torch.frexp(block_max) # torch.frexp(x) 把正浮点数拆成 mantissa * 2^exponent
    shared_exp = torch.where(block_max > 0, (exponent - 3).clamp(-127, 127), 0)
    block_scale = torch.exp2(shared_exp.to(w.dtype))

    # E8M0 只存指数。参考转换遇到超过 E2M1 范围的元素时会饱和到 ±6。
    restored = round_e2m1(blocks / block_scale) * block_scale # 量化反量化

    return restored.reshape_as(w), block_scale


def show_outlier_example() -> None:
    clean = torch.tensor([[0.5, 0.75, 1.0, 1.5]]).repeat(1, 16)
    outlier = clean.clone()
    outlier[0, 16] = 32

    # 位置 1、17、33 的原值都是 0.75。17 与离群值同属一个 MX 块，
    # 但在不同的 NV 块。33 位于两种格式的下一个块组。
    positions = [1, 17, 33]
    for name, w in (("clean", clean), ("outlier at index 0", outlier)):
        nv, nv_blocks, nv_global = nvfp4(w)
        mx, mx_blocks = mxfp4(w)
        print(f"{name}:")
        print(f"  NV global={nv_global.item():.6g}, E4M3 blocks={nv_blocks.flatten().tolist()}")
        print(f"  MX E8M0 blocks={mx_blocks.flatten().tolist()}")
        print(f"  W{positions}={w[0, positions].tolist()}")
        print(f"  NV{positions}={nv[0, positions].tolist()}")
        print(f"  MX{positions}={mx[0, positions].tolist()}")


def show_layer_error() -> None:
    path = Path(__file__).parents[1] / "weight_quant" / "down_proj_sample.pt"
    sample = torch.load(path, map_location="cpu", weights_only=True)
    w, bias = sample["weight"], sample["bias"]
    x = sample["validation"]["x"]
    ref = sample["validation"]["y_ref"]

    e4m3, scale = quantize_e4m3(w)
    nv, _, _ = nvfp4(w)
    mx, _ = mxfp4(w)
    for name, restored in (("E4M3", e4m3.float() * scale), ("NVFP4", nv), ("MXFP4", mx)):
        actual = x @ restored.T
        if bias is not None:
            actual += bias
        zeros = (restored == 0).float().mean().item()
        relerr = ((actual - ref).norm() / ref.norm()).item()
        print(f"{name}: zeros={zeros:.4%}, output relerr={relerr:.4%}")


if __name__ == "__main__":
    show_outlier_example()
    show_layer_error()
