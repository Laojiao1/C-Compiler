"""教学调度器：只形成计划，不执行模型、不修改 KV。

本步假设所有候选请求均已到达，KV 容量充足；
普通 Decode 每次只有一个已知但尚未计算 K/V 的输出 Token。
完成与重算反馈由独立的 CPU 执行模拟器负责；队列和容量由外层管理。
"""

from dataclasses import dataclass

from stage_1_paged_kv.paged_kv import RequestKV


@dataclass
class SchedulingRequest:
    """在已有 RequestKV 外附加调度信息，不改变原有块管理接口。"""

    request_id: str
    prompt_len: int
    kv: RequestKV
    # 已采样的输出数量，包括最新那个尚未经过前向、没有 K/V 的 Token。
    # 本步只记数量，不运行模型；后续执行模拟会负责更新它。
    num_generated_tokens: int = 0
    # 本步仅用输出数量作为终止条件，不模拟 EOS 或真实 Token 内容。
    max_new_tokens: int = 16
    # 完成后 KV 会被释放，有效长度回到 0，因此不能靠 KV 长度判断结束。
    finished: bool = False
    # 抢占前已有效计算的历史终点（不含该位置）。0 表示没有恢复任务。
    # 这不是输出计数；分块恢复期间仍需保留原来的完整终点。
    recompute_until: int = 0
    # 累计实际重算的位置数量；重复抢占造成的重复工作也要计入。
    num_recomputed_tokens: int = 0

    @property
    def num_pending_tokens(self) -> int:
        """已知 Token 总数减去已写入 KV 的位置数，不包含未来未知输出。"""
        return self.prompt_len + self.num_generated_tokens - self.kv.num_tokens

    @property
    def is_prefill(self) -> bool:
        """输入尚未全部计算时属于 Prefill；不是根据输出数量猜阶段。"""
        return self.kv.num_tokens < self.prompt_len

    @property
    def is_recomputing(self) -> bool:
        """有效 KV 尚未恢复到抢占前的终点，需要先补齐历史。"""
        return self.kv.num_tokens < self.recompute_until


@dataclass(frozen=True)
class ScheduledChunk:
    """一条执行指令：处理 [start_pos, start_pos + num_tokens) 的位置。"""

    request_id: str
    phase: str
    start_pos: int
    num_tokens: int


def plan_step(
    requests: list[SchedulingRequest],
    max_num_batched_tokens: int,
    max_num_seqs: int,
) -> list[ScheduledChunk]:
    """Decode 优先，再以剩余预算安排重算或分块 Prefill。

    输入约定：请求 ID 唯一，prompt_len > 0，进度和已生成数量有效。
    已完成请求允许留在输入列表中，但不参与选择。
    输出每条请求至多一项；同类请求保持输入顺序。
    两个上限必须 > 0，否则抛出 ValueError。

    注意：这是只读的候选计划，不是已经执行成功的调度结果。
    不调用 ensure_capacity，不更新 kv.num_tokens，不修改输入列表。
    后续必须加入真实容量检查和执行反馈，才能形成完整调度步骤。
    """
    if max_num_batched_tokens <= 0 or max_num_seqs <= 0:
        raise ValueError("Token budget 和序列上限必须大于 0")

    active = [
        request
        for request in requests
        if not request.finished and request.num_pending_tokens > 0
    ]

    decode = [request for request in active
              if not request.is_prefill and not request.is_recomputing]
    bulk = [request for request in active
            if request.is_prefill or request.is_recomputing]

    remaining_tokens = max_num_batched_tokens
    plan: list[ScheduledChunk] = []

    for request in decode + bulk:
        # Token 数与序列数是独立限制，任意一个用尽都必须停止。
        if remaining_tokens == 0 or len(plan) >= max_num_seqs:
            break

        if request.is_recomputing:
            # 本轮只恢复已经算过的历史，不顺便处理最新的待算输出。
            count = min(request.recompute_until - request.kv.num_tokens, remaining_tokens)
            phase = "recompute"
        elif request.is_prefill:
            # Prefill 可以切块：只安排剩余输入与剩余预算中较小的数量。
            count = min(request.num_pending_tokens, remaining_tokens)
            phase = "prefill"
        else:
            # 本步只支持普通逐 Token Decode，不涉及推测解码或历史重算。
            count = 1
            phase = "decode"

        plan.append(
            ScheduledChunk(
                request_id=request.request_id,
                phase=phase,
                # 从已有有效 KV 的末尾接着计算，而不是从预留容量末尾开始。
                start_pos=request.kv.num_tokens,
                num_tokens=count,
            )
        )

        # 这里只扣减局部预算，不改变请求的计算进度或实际物理块。
        remaining_tokens -= count

    return plan
