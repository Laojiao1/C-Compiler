import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str                              # 本地模型目录
    max_num_batched_tokens: int = 16384     # 单步最多处理的 token 总数（token Budget）
    max_num_seqs: int = 512                 # 单步最多并发的序列数
    max_model_len: int = 4096               # 支持的最大上下文长度
    gpu_memory_utilization: float = 0.9     # 允许占用的显存比例
    tensor_parallel_size: int = 1           # 张量并行 GPU 数
    enforce_eager: bool = False             # True 表示禁用 CUDA Graph
    hf_config: AutoConfig | None = None     # 延迟加载的 HuggingFace 配置
    eos: int = -1                           # EOS token id（初始化后由 tokenizer 填入）
    kvcache_block_size: int = 256           # 每个 KV 块容纳的 token 数
    num_kvcache_blocks: int = -1            # KV 块总数（运行时按显存推算）

    def __post_init__(self):
        assert os.path.isdir(self.model) # 必须是存在本地的目录
        assert self.kvcache_block_size % 256 == 0 # 块大小需要是 256 的倍数
        assert 1 <= self.tensor_parallel_size <= 8 # TP 度在 1~8 之间
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
