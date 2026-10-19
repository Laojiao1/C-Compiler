from typing import Callable, List, Literal, Optional, Tuple

import torch

from model import GPT2BlockReference, GPT2ReferenceModel


CacheMode = Literal["cat", "prealloc"]
KVCache = Tuple[torch.Tensor, torch.Tensor]
KVCacheList = List[KVCache]
AttentionImpl = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor]


# 把大 GEMM 计算出来的单一稠密张量，拆分并重组为适合执行并行多头注意力（MHA）的标准内存布局
def _split_qkv(
    qkv: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 把 [B, T, 3 * hidden] 拆成三个 [B, heads, T, head_dim]
    batch_size, seq_len, _ = qkv.shape
    # [B, T, 2304] -> [B, T, 3, 12, 64]
    # 沿着 dim=2 剥离，每个张量形状均为 [B, T, 12, 64]
    qkv = qkv.view(batch_size, seq_len, 3, num_heads, head_dim)
    query, key, value = qkv.unbind(dim=2)

    # [B, T, 12, 64] -> [B, 12, T, 64]
    # 后续 Attention 的矩阵乘法使用 [B, heads, T, D] 布局
    return (
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
    )


# _split_qkv 的逆操作算子，
# 把 [B, heads, T, head_dim] 合并回 [B, T, hidden_size]
def _merge_heads(
    attention_output: torch.Tensor,
    hidden_size: int,
) -> torch.Tensor:
    # 输入的 attention_output 形状为：[B, H, T, D]，
    # 但后续的线性层 c_proj（形状 [768, 768]）期望输入的形状是 [B, T, 768] 
    # 即每个 Token 的 768 维特征必须完整、连续地排在最后一维
    batch_size, _, seq_len, _ = attention_output.shape

    # 1. 换轴：[B, H, T, D] -> [B, T, H, D]
    attention_output = attention_output.transpose(1, 2).contiguous()

    # 2. 展平后两维：[B, T, H, D] -> [B, T, H * D]，即 [B, T, 768]
    return attention_output.view(batch_size, seq_len, hidden_size)


# 把 Prefill 和 Decode，统一抽离到了同一个标准注意力函数中
def _run_attention(
    block: GPT2BlockReference,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    apply_causal_mask: bool,
    attention_impl: Optional[AttentionImpl] = None,
) -> torch.Tensor:
    # 完成一个 Block 中的 Attention 子层，不包含 residual。
    # query: [B, H, query_len, D]
    # key.transpose(-1, -2): [B, H, D, key_len]
    # scores: [B, H, query_len, key_len]
    _, _, query_len, _ = query.shape
    _, _, key_len, _ = key.shape

    if attention_impl is None:
        scores = query @ key.transpose(-1, -2)
        scores = scores / (block.attn.head_dim ** 0.5)

        if apply_causal_mask:
            # Prefill 时 Query 有多个位置，需要屏蔽未来 Key。
            causal_mask = block.attn.bias[:, :, :query_len, :key_len]
            scores = scores.masked_fill(
                ~causal_mask,
                torch.finfo(scores.dtype).min,
            )

        # Decode 单步时不需要 mask：Cache 中没有未来 token。
        probabilities = torch.softmax(scores, dim=-1)
        attention_output = probabilities @ value
    else:
        # OperatorEngine 要求连续内存，而 Q/K/V 是 transpose 得到的视图。
        # 把复制留在计时区域内，测到的端到端结果才包含真实接入成本。
        attention_output = attention_impl(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            apply_causal_mask,
        )

    # 3. 合并多头回统一隐层: [B, query_len, hidden_size]
    attention_output = _merge_heads(
        attention_output,
        block.attn.hidden_size,
    )

    # 4. 线性输出投影 + Dropout
    return block.attn.resid_dropout(block.attn.c_proj(attention_output))


# 解耦 Prefill 与 Decode 在嵌入阶段的不同调用形态
def _embed(
    model: GPT2ReferenceModel,
    input_ids: torch.Tensor, 
    position_ids: torch.Tensor,
) -> torch.Tensor:
    # wte 词嵌入查表：输入 Token ID，通过索引查找此表矩阵，得到语义向量表示
    hidden_states = model.transformer.wte(input_ids)
    # wpe 位置嵌入查表：输入绝对位置索引，通过索引查找位置矩阵，得到向量位置表示
    # 将 语义信息 与 时序位置信息 叠加
    hidden_states = hidden_states + model.transformer.wpe(position_ids)
    # dropout
    return model.transformer.drop(hidden_states)


def _prefill_block(
    block: GPT2BlockReference,
    hidden_states: torch.Tensor,
    mode: CacheMode,
    max_seq_len: int,
    attention_impl: Optional[AttentionImpl] = None,
) -> Tuple[torch.Tensor, KVCache]:
    """处理一个完整 prompt，并建立该层的初始 KV Cache。"""
    batch_size, seq_len, _ = hidden_states.shape
    residual = hidden_states
    normalized = block.ln_1(hidden_states)
    qkv = block.attn.c_attn(normalized)
    query, key, value = _split_qkv(
        qkv,
        block.attn.num_heads,
        block.attn.head_dim,
    )

    attention_output = _run_attention(
        block,
        query,
        key,
        value,
        apply_causal_mask=True,
        attention_impl=attention_impl,
    )
    hidden_states = residual + attention_output

    residual = hidden_states
    hidden_states = block.ln_2(hidden_states)
    hidden_states = residual + block.mlp(hidden_states)

    if mode == "cat":
        # cat 路径直接把当前已有的 K/V 保存下来。
        cache = (key, value)
    else:
        # 预分配路径一次开好最大空间，只把 prompt 写到前 seq_len 个位置。
        key_cache = torch.empty(
            batch_size,
            block.attn.num_heads,
            max_seq_len,
            block.attn.head_dim,
            dtype=key.dtype,
            device=key.device,
        )
        value_cache = torch.empty_like(key_cache)
        key_cache[:, :, :seq_len, :].copy_(key)
        value_cache[:, :, :seq_len, :].copy_(value)
        cache = (key_cache, value_cache)

    return hidden_states, cache


def _decode_block(
    block: GPT2BlockReference,
    hidden_states: torch.Tensor,
    cache: KVCache,
    position: int,
    mode: CacheMode,
) -> Tuple[torch.Tensor, KVCache]:
    """处理一个新 token，并更新当前层的 KV Cache。"""
    residual = hidden_states
    normalized = block.ln_1(hidden_states)
    qkv = block.attn.c_attn(normalized)
    query, key_new, value_new = _split_qkv(
        qkv,
        block.attn.num_heads,
        block.attn.head_dim,
    )

    old_key, old_value = cache

    if mode == "cat":
        # ====== 路径 B：torch.cat Cache ======
        # 每一步都生成新的结果并复制旧 Cache，
        # 这正是我们要测量的 O(T^2) 累计复制成本
        # 张量沿序列维（dim=2）拼接，变成 [B, H, t + 1, D]
        key = torch.cat((old_key, key_new), dim=2)
        value = torch.cat((old_value, value_new), dim=2)
        new_cache = (key, value)
    else:
        # ====== 路径 C：预分配 Cache ======
        # 直接写入当前位置，不复制历史数据。
        # 1. 原地写入：把新产生的单步 KV 填入当前 position 槽位
        old_key[:, :, position : position + 1, :].copy_(key_new)
        old_value[:, :, position : position + 1, :].copy_(value_new)
        # 2. 视图切片：截取截至当前已填充的有效前缀
        key = old_key[:, :, : position + 1, :]
        value = old_value[:, :, : position + 1, :]
        new_cache = (old_key, old_value)

    hidden_states = residual + _run_attention(
        block,
        query, # [B, H, 1, D]
        key, # [B, H, t + 1, D]
        value, # [B, H, t + 1, D]
        apply_causal_mask=False, # 为什么是 False？因为没有未来 Token
    )

    residual = hidden_states
    hidden_states = block.ln_2(hidden_states)
    hidden_states = residual + block.mlp(hidden_states)
    return hidden_states, new_cache


# ===== 路径 A：无缓存 Decode =====
# 实际工业中不会采用这种方法，仅适用于基线
@torch.no_grad()
def decode_without_cache(
    model: GPT2ReferenceModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    每一步都重新计算完整序列，作为正确性基线。
    这条路径不保存任何历史 K/V，因此最容易理解，但会重复计算历史 token。
    """

    sequence = input_ids # 初始化序列为 Prompt
    step_logits: List[torch.Tensor] = []

    for _ in range(max_new_tokens):
        # 1. 全序列前向传播：输入当前已有的整个序列
        logits = model(sequence) # 输出形状: [B, T_current, 50257]

        # 2. 提取最后一个位置的预测分布
        next_logits = logits[:, -1, :] # 提取第 -1 个位置: [B, 50257]

        # 3. 贪婪解码（Greedy Search）选出最高概率的 Token
        next_token = next_logits.argmax(dim=-1, keepdim=True) # [B, 1]

        # 4. 记录当前步的 Logits 并拼接到序列尾部
        step_logits.append(next_logits)
        sequence = torch.cat((sequence, next_token), dim=1) # 序列长度 + 1

    return sequence, step_logits


# ===== 路径 B/C：带 KV Cache 的 Decode ======
@torch.no_grad()
def decode_with_cache(
    model: GPT2ReferenceModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    mode: CacheMode,
    prefill_attention_impl: Optional[AttentionImpl] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    mode="cat"：路径 B，每一步用 torch.cat 生成新的 Cache

    mode="prealloc"：路径 C，提前分配最大 Cache 并原地写入当前位置

    prefill_attention_impl：可选的 Prefill Attention 实现。Decode 阶段仍走
    当前 PyTorch Attention，因为 OperatorEngine 不接受 Q/K/V 不同长度。
    """
    if mode not in ("cat", "prealloc"):
        raise ValueError(f"unsupported cache mode: {mode}")

    # 全局上下文准备：
    batch_size, prompt_len = input_ids.shape
    # 为后续 prealloc 模式指定静态开辟的最大容量上限
    max_seq_len = prompt_len + max_new_tokens
    # 生成 [0, 1, ..., prompt_len - 1]，对 Prompt 进行标准编码
    position_ids = torch.arange(
        prompt_len,
        device=input_ids.device,
    ).unsqueeze(0)

    hidden_states = _embed(model, input_ids, position_ids)
    caches: KVCacheList = []

    # 逐层建立 Cache 池：
    # 遍历全部 12 个 Block，每个 Block 都会产出一个属于该层的 cache（包含对应的 Key 和 Value）
    # 将这 12 层的 Cache 按顺序收集进 caches: List[Tuple[Tensor, Tensor]] 中
    # 此时，Prompt 的全部历史信息已经被固化在了各个层独立的显存空间中
    for block in model.transformer.h:
        hidden_states, cache = _prefill_block(
            block,
            hidden_states,
            mode,
            max_seq_len,
            attention_impl=prefill_attention_impl,
        )
        caches.append(cache)

    # 首字生成（Time-To-First-Token,TTFT）
    hidden_states = model.transformer.ln_f(hidden_states)
    # 在 Prefill 结束后，hidden_states 形状是 [B, prompt_len, 768]
    next_logits = model.lm_head(hidden_states[:, -1, :])
    # 只有 Prompt 的最后一个 Token 位置（-1）的隐藏状态融合了整段 Prompt 的所有上下文，直接将其送入 lm_head 即可得到第一个生成词（第 1 个 New Token）的预测分布
    # 将这个首字 Logits 作为第 0 步结果放入 step_logits
    step_logits = [next_logits]
    sequence = input_ids
    # 用 argmax 贪婪选取概率最高的 Token，准备递交给接下来的增量循环
    next_token = next_logits.argmax(dim=-1, keepdim=True)

    # Decode 增量自回归循环 （Time-Per-Output-Token，TPOT）
    for step in range(max_new_tokens):
        sequence = torch.cat((sequence, next_token), dim=1)

        # 1. 绝对位置偏移推进
        position = prompt_len + step
        token_position = torch.full(
            (batch_size, 1),
            position,
            dtype=torch.long,
            device=input_ids.device,
        )
        hidden_states = _embed(
            model,
            next_token,
            token_position,
        )

        # 2. 逐层前向传播与 Cache 更新
        updated_caches: KVCacheList = []
        for block, cache in zip(model.transformer.h, caches):
            hidden_states, cache = _decode_block(
                block,
                hidden_states,
                cache,
                position,
                mode,
            )
            updated_caches.append(cache)
        caches = updated_caches

        # 3. 产出新词并更新
        hidden_states = model.transformer.ln_f(hidden_states)
        next_logits = model.lm_head(hidden_states[:, -1, :])

        # 4. 边界截断保护：最后一轮只需要生成 token，不需要再为下一轮保存 logits。
        if step + 1 < max_new_tokens:
            step_logits.append(next_logits)
            next_token = next_logits.argmax(dim=-1, keepdim=True)

    return sequence, step_logits
