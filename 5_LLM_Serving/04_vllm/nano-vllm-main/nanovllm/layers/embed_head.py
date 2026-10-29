# 词表并行 Embedding 与 LMHead
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


"""
词表并行：每张卡只持有词表的一段
"""
class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    # 要处理「某个 token 不属于本卡」的情况
    # 每张卡都查一次，但只有「负责该 token 的卡」查出非零向量，其余卡输出 0，all_reduce 一加就得到完整结果
    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # 本卡负责的 token
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 平移到本地下标，越界的先置0
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y # 不属于本卡的行清零
            dist.all_reduce(y) # 求和：每个 token 的向量只有一卡非零
        return y


"""
负责把 hidden state 投影成词表 logits
"""
# 注意这里采用的是 gather 而不是 reduce：因为词表是按照「输出」维度切分的，最终应该拼接起来
class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1 # 每条序列最后一个位置
            x = x[last_indices].contiguous() # 只取最后一个 token
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # 各卡算出词表的一段，gather 到 rank0 再拼接成完整的 logits
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
