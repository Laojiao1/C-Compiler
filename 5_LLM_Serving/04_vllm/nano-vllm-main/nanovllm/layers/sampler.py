# 温度采样
import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 温度缩放，分别对两行 Logits 除以各自的温度
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        # 计算得到词表概率分布 probs，Shape 为 [2, vocab_size]。
        probs = torch.softmax(logits, dim=-1)

        # 利用除以指数分布噪声再求 argmax 的技巧，在 GPU 上实现极其高效、并行的数学等价无偏随机采样
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)

        # 返回结果：sample_tokens 是一个 Shape 为 [2] 的 GPU 张量
        return sample_tokens
