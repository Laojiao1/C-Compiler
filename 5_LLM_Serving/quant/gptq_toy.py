import torch


def quantize_int4(weight: torch.Tensor) -> torch.Tensor:
    # scale 固定为 1，整数范围为 [-7, 7]。
    # 返回量化再反量化后的浮点值，供矩阵乘法使用。
    return weight.round().clamp(-7, 7)


def main() -> None:
    x = torch.tensor([[1.0, 1.0], [1.0, 0.8]], dtype=torch.float64)
    w = torch.tensor([2.4, 3.4], dtype=torch.float64)
    y_ref = x @ w

    # RTN：两个权重各自舍入。
    w_rtn = quantize_int4(w)

    # GPTQ 两列演示：先量化第一列，记录它损失的权重值。
    w0_quantized = quantize_int4(w[0])
    first_error = w[0] - w0_quantized

    # 求 delta，使 ||first_error * x0 - delta * x1||² 最小。
    # x0、x1 分别是两个输入通道在所有 token 上的激活。
    x0, x1 = x[:, 0], x[:, 1]
    delta = first_error * torch.dot(x1, x0) / torch.dot(x1, x1)

    # 第二列先补偿，再舍入到与 RTN 相同的 INT4 网格。
    w1_adjusted = w[1] + delta
    w_gptq = torch.stack((w0_quantized, quantize_int4(w1_adjusted)))

    print(f"first_error = {first_error.item():.6f}")
    print(f"delta = {delta.item():.6f}")
    print(f"second weight before rounding = {w1_adjusted.item():.6f}")
    print(f"reference Y = {y_ref.tolist()}")

    for name, weight in (("RTN", w_rtn), ("GPTQ toy", w_gptq)):
        y = x @ weight
        relerr = (y - y_ref).norm() / y_ref.norm()
        print(f"{name}: W={weight.tolist()}, Y={y.tolist()}, relerr={relerr.item():.4%}")


if __name__ == "__main__":
    main()