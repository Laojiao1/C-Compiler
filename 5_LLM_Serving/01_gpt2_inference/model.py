import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import GPT2Config


TensorOrOutput = Union[
    torch.Tensor,
    Tuple[torch.Tensor, List[torch.Tensor]],
]


class GPT2Conv1D(nn.Module):
    """GPT-2 中名为 Conv1D 的线性层。

    它实际执行 ``x @ weight + bias``，其中 weight 的形状是
    ``[input_dim, output_dim]``。这和 nn.Linear 的权重布局相反。

    这个名字容易造成误解：这里没有做图像卷积，也没有滑动窗口。
    GPT-2 只是把输入最后一维的 hidden state 映射到新的特征维度。
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入的最后一维做 GPT-2 风格的线性变换。"""
        input_shape = x.shape

        # x 可能是 [B, T, C]，而 addmm 只接受二维矩阵。
        # 把 B 和 T 暂时合并，不会改变最后一维 C 的含义。
        x_2d = x.view(-1, input_shape[-1])

        # GPT-2 Conv1D 的 weight 是 [C_in, C_out]，
        # 所以这里直接做 x @ weight，而不是 x @ weight.T。
        output = torch.addmm(self.bias, x_2d, self.weight)

        # 把展平的维度恢复回来，例如：
        # [B*T, C_out] -> [B, T, C_out]。
        return output.view(*input_shape[:-1], self.weight.shape[1])


class NewGELU(nn.Module):
    """GPT-2 使用的 gelu_new 近似公式。

    不能直接用任意 GELU 替代它。不同 GELU 公式的差异虽然很小，
    经过 12 层 Transformer 累积后，最终 logits 仍然会出现偏差。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coefficient = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (
            1.0 + torch.tanh(
                coefficient * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class GPT2AttentionReference(nn.Module):
    """GPT-2 的多头因果自注意力，负责计算序列中每个词与其它词之间的关联度。

    当前只实现 Prefill：Q、K、V 的序列长度相同，不处理 KV Cache。
    Decode 阶段的 past K/V 会在后续 decode.py 中单独加入。
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()

        self.num_heads = config.n_head # 头数 H，GPT-2 Small 中为 12
        self.hidden_size = config.n_embd # 隐藏层维度 D，GPT-2 Small 中为 768
        self.head_dim = self.hidden_size // self.num_heads # 每个头的维度 d = 768 // 12 = 64

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden size must be divisible by num_heads")

        # 1. QKV 算子融合：将三者拼成一个 [B, T, 2304] 的大权重矩阵，单次 GEMM 一步到位
        # [B, T, 768] -> [B, T, 2304]
        self.c_attn = GPT2Conv1D(self.hidden_size, 3 * self.hidden_size)

        # 把所有头的特征做一次全连接线性融合，打通跨头之间的信息通道
        self.c_proj = GPT2Conv1D(self.hidden_size, self.hidden_size)
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # causal_mask[i, j] 为 True，表示位置 i 可以读取位置 j。
        # 下三角意味着第 i 个 token 只能看到自己和更早的 token。
        causal_mask = torch.tril(
            torch.ones(config.n_ctx, config.n_ctx, dtype=torch.bool)
        )
        self.register_buffer(
            "bias",
            causal_mask.view(1, 1, config.n_ctx, config.n_ctx),
            persistent=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # 1. 融合投影：输出为 [B, T, 3 * C]，其中 C = 768
        qkv = self.c_attn(hidden_states)

        # 2. 升维拆解：[B, T, 3, H, D]，其中 H = 12, D = 64
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        # 3. 沿 dim = 2 解出独立的 Q, K, V，各自形状均为 [B, T, H, D]
        query, key, value = qkv.unbind(dim=2)

        # 4. 调整多头维度到第 1 维：[B, T, H, D] -> [B, H, T, D]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # [B, H, T, D] @ [B, H, D, T] -> [B, H, T, T]
        # 最后两个维度做矩阵乘法，每个 Query 都和所有 Key 做点积。
        attention_scores = query @ key.transpose(-1, -2)

        # 缩放因子
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        # 动态截取序列实际长度，避免使用完整的 max_position 对矩阵做无效计算
        causal_mask = self.bias[:, :, :seq_len, :seq_len]

        # 屏蔽未来信息
        attention_scores = attention_scores.masked_fill(
            ~causal_mask,
            torch.finfo(attention_scores.dtype).min,
        )

        # 对行归一化：[B, H, T, T]，每一行表示一个 Query 对序列中所有 Key 的权重分配
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        # 用注意力概率对 V 做加权求和：
        # [B, H, T, T] @ [B, H, T, D] -> [B, H, T, D]
        attention_output = attention_probs @ value

        # 还原维度顺序：[B, H, T, D] -> [B, T, H * D]。
        # transpose 后内存通常不连续，view 前必须 contiguous
        attention_output = attention_output.transpose(1, 2).contiguous()
        # 压平多头回原始隐层维度：[B, T, H * D] 即 [B, T, 768]
        attention_output = attention_output.view(
            batch_size,
            seq_len,
            self.hidden_size,
        )

        # 4. 输出线性变换与残差 Dropout
        attention_output = self.c_proj(attention_output)
        return self.resid_dropout(attention_output)


class GPT2MLPReference(nn.Module):
    """GPT-2 的两层 MLP：768 -> 3072 -> 768。

    中间层扩大到 4 倍 hidden size，增加非线性变换的表达能力。

    如果说前面的 Attention 模块负责 在序列维度上做信息交互（跨 Token 聚合），
    那么 MLP 模块则负责 在特征维度上做非线性变换与知识存储（单 Token 内提取）
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        # GPT-2 Small 中 768 * 4 = 3072。
        intermediate_size = 4 * config.n_embd
        self.c_fc = GPT2Conv1D(config.n_embd, intermediate_size)
        self.activation = NewGELU()
        self.c_proj = GPT2Conv1D(intermediate_size, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 1. 升维投影: [B, T, 768] -> [B, T, 3072]
        hidden_states = self.c_fc(hidden_states)

        # 2. 逐元素非线性激活 (Shape 不变): [B, T, 3072]
        hidden_states = self.activation(hidden_states)

        # 3. 降维投影: [B, T, 3072] -> [B, T, 768]
        hidden_states = self.c_proj(hidden_states)

        # 4. 残差 Dropout (训练防过拟合，推理为 Identity)
        return self.dropout(hidden_states)


class GPT2BlockReference(nn.Module):
    """一个 GPT-2 Pre-LN Transformer Block。

    GPT-2 的两个 residual 分支都保留原始输入，
    这样 Attention 或 MLP 学习的是对输入的增量修正。
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2AttentionReference(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLPReference(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # ------ 阶段 1：自注意力处理 ------
        residual = hidden_states  # 1. 暂存残差基底
        hidden_states = self.ln_1(hidden_states)  # 2. 归一化输入
        hidden_states = residual + self.attn(
            hidden_states
        )  # 3. 计算增量并相加（更新残差主干）

        # ------ 阶段 2：前馈网络处理 ------
        residual = hidden_states  # 4. 暂存新的残差基底
        hidden_states = self.ln_2(hidden_states)  # 5. 归一化输入
        return residual + self.mlp(hidden_states)  # 6. 计算增量并相加，输出给下一层


class GPT2ReferenceModel(nn.Module):
    """GPT-2 Small 的手写 Prefill 参考模型。

    当前 forward 只处理 ``input_ids -> logits``，暂时不加入 KV Cache。
    KV Cache 会在原路线规定的 decode.py 中单独实现。
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config

        # 保留 Hugging Face 的层级命名，方便直接加载 state_dict。
        # wte 是 token embedding，wpe 是 learned position embedding。
        # 可学习的位置编码（Learned Positional Embedding）：不同于原始 Transformer使用的无参数正弦/余弦固定编码（Sinusoidal），GPT-2 采用纯粹随机初始化并可学习的参数表。这也是后来很多现代 LLM（直到 RoPE 出现前）的主流做法

        self.transformer = nn.Module()

        # wte (Word Token Embedding)
        # shape [50257, 768]。负责将 Token ID 映射为 768 维语义向量
        self.transformer.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # wpe (Word Position Embedding)：
        # 形状：[1024, 768]（GPT-2 最大上下文为 1024）
        self.transformer.wpe = nn.Embedding(config.n_positions, config.n_embd)

        self.transformer.drop = nn.Dropout(config.embd_pdrop)

        # 只有注册进 ModuleList，其中的 12 个 Block 及其包含的子参数才会被 PyTorch 识别为子模块，正确响应 .to(device)、.eval() 以及梯度更新。
        self.transformer.h = nn.ModuleList(
            [GPT2BlockReference(config) for _ in range(config.n_layer)]
        )

        # 残差主干上的数值在经过 12 层的不断累加后，方差会持续累积变大。在最终投影回词表概率之前，必须由 ln_f 将总线上的特征重新拉回均值为 0、方差为 1 的稳定尺度，否则进入 lm_head 后的 Logits 幅度会失控
        self.transformer.ln_f = nn.LayerNorm(
            config.n_embd,
            eps=config.layer_norm_epsilon,
        )

        # 这块是一个很重要的设计技巧：通过点积计算，可以反映前的上下文隐层表征 x 与词表中的候选词向量 W 在方向上的相似度，哪个词的向量和当前思考出来的特征向量 x 越接近、越同向，它的点积得分（Logit）就越高，Softmax 之后被选为下一个词的概率就越大
        # 不设置 bias 的原因在于防止全局词频偏置破坏上下文语义，强迫模型必须纯粹依据当前的隐层语义特征 x 与词嵌入的相似度来判断下一个词，泛化能力更强
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 通过引用赋值，两处指针指向同一块底层显存缓冲区，节省显存的同时加快了词向量表示的学习收敛速度
        self.lm_head.weight = self.transformer.wte.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> TensorOrOutput:
        _, seq_len = input_ids.shape

        if position_ids is None:
            # Prefill 从位置 0 开始，位置索引为 [0, 1, ..., seq_len - 1]。
            # Decode 时会传入带有历史偏移的 position_ids。
            position_ids = torch.arange(
                seq_len,
                device=input_ids.device,
            ).unsqueeze(0)

        # token embedding 和 position embedding 都是 [B, T, HIDDEN]
        # wte 根据 Token ID 查表，得到语义向量，形状为 [B, T, 768]
        hidden_states = self.transformer.wte(input_ids)
        # wpe 根据位置索引查表，得到位置向量，形状为 [1, T, 768]
        hidden_states = hidden_states + self.transformer.wpe(position_ids)
        hidden_states = self.transformer.drop(hidden_states)

        # hidden_states 沿着 12 个 Block 构成的残差总线一路前向传播，依次经历 12 次 Self-Attention 跨词聚合与 12 次 MLP 知识注入
        all_hidden_states: List[torch.Tensor] = [hidden_states]
        for block in self.transformer.h:
            hidden_states = block(hidden_states)
            all_hidden_states.append(hidden_states)

        # 离开 Block 之后，通过 ln_f 将累积方差重新归一化为均值 0、方差 1
        hidden_states = self.transformer.ln_f(hidden_states)
        all_hidden_states.append(hidden_states)

        # 输出每个位置对整个词表的 logits：
        # [B, T, 768] -> [B, T, 50257]。
        logits = self.lm_head(hidden_states)

        if return_hidden_states:
            return logits, all_hidden_states
        
        return logits
