from pathlib import Path
import torch

QMAX = 7


def quantize(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (weight / scale).round().clamp(-QMAX, QMAX) * scale

# 关键代码:
def GPTQ(weight: torch.Tensor, x_cal: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    gram = x_cal.T @ x_cal # 格拉姆矩阵
    damping = 0.01 * gram.diagonal().mean() # 阻尼系数
    gram.diagonal().add_(damping) # 给格拉姆矩阵的对角线元素加上阻尼系数，防止矩阵奇异不可逆

    # 这里的目的是为了快速求 gram 的逆（正定矩阵的逆矩阵）：先通过 cholesky 分解然后对其求逆，可以减少计算量
    chol = torch.linalg.cholesky(gram)
    gram_inv = torch.cholesky_inverse(chol)

    # 利用线性代数的特殊性质，把对阵正定矩阵分解成上三角矩阵U的乘积时，U的每一行在经过对角线的归一化后，正好等于后面需要补偿的系数
    upper = torch.linalg.cholesky(gram_inv, upper=True)

    working = weight.clone()
    quantized = torch.empty_like(weight)

    for col in range(weight.shape[1]):
        # 一列对应一个输出通道，一次处理该通道上的全部 N 个权重
        current = working[:, col] # 取权重
        quantized[:, col] = quantize(current, scale) # 量化反量化

        column_error = current - quantized[:, col] # 计算误差
        coefficients = upper[col, col + 1:] / upper[col, col] # 从 Cholesky 上三角矩阵中提取误差分摊比例

        working[:, col + 1: ] -= column_error[:, None] * coefficients # 误差补偿到后续的列

    return quantized


# 误差对比
def relative_output_error(
    x: torch.Tensor, quantized: torch.Tensor, weight: torch.Tensor
) -> float:
    # 只取 32 个输入通道。参考输出要用同一个切片重算，因为采集文件里的 y_ref 包含全部 8960 个输入通道的贡献
    reference = x @ weight.T
    output = x @ quantized.T
    return ((output - reference).norm() / reference.norm()).item()


def main() -> None:
    path = Path(__file__).with_name("down_proj_sample.pt")
    sample = torch.load(path, map_location="cpu", weights_only=True)

    # 取同一层的前 64 个输出通道和前 32 个输入通道。
    # 两组激活也切相同的输入通道，确保 X 的列与 W 的列一一对应。
    # 用 float64 完成这个小实验，减少求逆和 Cholesky 分解的舍入影响。
    weight = sample["weight"][:64, :32].double()   # [64, 32]
    x_cal = sample["calibration"]["x"][:, :32].double()  # [21, 32]
    x_val = sample["validation"]["x"][:, :32].double()   # [25, 32]

    # 从原始权重切片计算一个全局 scale。两条路径共用这个 scale，
    # 所以逐列补偿只能改变选中的量化档位，网格本身保持固定。
    max_abs = weight.abs().amax()
    scale = torch.where(max_abs > 0, max_abs / QMAX, torch.ones_like(max_abs))

    # RTN 直接把每个原始权重舍入到最近的网格点。
    # GPTQ 根据校准激活，先补偿尚未量化的权重，再逐列舍入。
    w_rtn = quantize(weight, scale)
    w_gptq = GPTQ(weight, x_cal, scale)

    # 两边用同一个 scale，反量化值不同就表示选中的 INT4 档位不同。
    changed = (w_rtn != w_gptq).sum().item()
    print(f"W={tuple(weight.shape)}, scale={scale.item():.6e}")
    print(f"Different INT4 levels: {changed}/{weight.numel()}")

    # 校准集用于构造 Gram。验证集只用于评估已量化的权重，用来观察基于校准激活计算的补偿能否迁移到另一组输入
    for split, x in (("calibration", x_cal), ("validation", x_val)):
        rtn_err = relative_output_error(x, w_rtn, weight)
        gptq_err = relative_output_error(x, w_gptq, weight)
        print(f"{split}: RTN={rtn_err:.4%}, GPTQ={gptq_err:.4%}")


if __name__ == "__main__":
    main()