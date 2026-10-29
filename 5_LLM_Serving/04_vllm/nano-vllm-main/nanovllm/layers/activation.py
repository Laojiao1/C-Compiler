# SiluAndMul（SwiGLU）
import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    # 算子融合：把输入投影成两份：一份当门控过 SiLU，另一份当数据，再逐元素相乘
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
