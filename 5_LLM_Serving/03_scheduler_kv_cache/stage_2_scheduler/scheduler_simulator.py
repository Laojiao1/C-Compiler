"""CPU 执行模拟：验证进度、输出数量与 KV 引用，不执行真实模型。

K/V 使用位置编号构造占位数值；不会计算 Attention、logits 或真实采样。
每个 chunk 单独提交：失败时该请求不推进，不保证整批事务回滚。
容量压力下怎样形成可执行 batch，留到后续容量感知调度处理。
"""

import torch

from stage_1_paged_kv.paged_kv_store import PagedKVStore
from stage_2_scheduler.scheduler import ScheduledChunk, SchedulingRequest


def _validate_chunk(
    request: SchedulingRequest,
    chunk: ScheduledChunk,
    store: PagedKVStore,
) -> None:
    """在任何写入前拒绝错配、过期或不符合普通生成流程的指令。"""
    if request.finished:
        raise ValueError("请求已经结束，不能继续执行")
    if request.prompt_len <= 0 or request.max_new_tokens <= 0:
        raise ValueError("输入长度与输出上限必须大于 0")
    if not 0 <= request.num_generated_tokens < request.max_new_tokens:
        raise ValueError("活动请求的已生成数量不合法")
    if request.kv.pool is not store.pool or request.kv.block_size != store.block_size:
        raise ValueError("请求必须使用执行存储对应的块池与块大小")
    if chunk.request_id != request.request_id:
        raise ValueError("执行指令与请求 ID 不匹配")
    if chunk.start_pos != request.kv.num_tokens:
        raise ValueError("指令起点与当前有效 KV 长度不匹配，可能重复执行了旧计划")
    if chunk.num_tokens <= 0 or chunk.num_tokens > request.num_pending_tokens:
        raise ValueError("执行数量必须位于待计算位置范围内")

    if request.is_recomputing:
        if chunk.phase != "recompute":
            raise ValueError("恢复期间只能执行历史重算")
        if chunk.start_pos + chunk.num_tokens > request.recompute_until:
            raise ValueError("本项不能越过历史恢复终点")
    elif request.is_prefill:
        if request.num_generated_tokens != 0:
            raise ValueError("生成后丢失 KV 时必须先登记恢复终点")
        if chunk.phase != "prefill":
            raise ValueError("输入尚未完成时只能执行 Prefill")
    else:
        if chunk.phase != "decode" or chunk.num_tokens != 1:
            raise ValueError("普通 Decode 必须恰好处理一个位置")
        if request.num_pending_tokens != 1:
            raise ValueError("普通 Decode 应恰有一个待计算的已采样输出")


def _finish_recompute(request: SchedulingRequest, chunk: ScheduledChunk) -> bool:
    """重算 K/V 已成功写入后，更新重算账目；不得新增或重发输出。"""
    # 本函数只在 store.append 成功后调用，因此这些位置确实已重建。
    request.num_recomputed_tokens += chunk.num_tokens

    # 分块恢复中途不能清除终点；只有恢复完整段旧历史后才能继续普通生成。
    if request.kv.num_tokens == request.recompute_until:
        request.recompute_until = 0

    # 重建旧历史没有新增输出，不能增加 num_generated_tokens，
    # 也不能向外报告又产生了一个输出 Token。
    return False


def execute_chunk(
    request: SchedulingRequest,
    chunk: ScheduledChunk,
    store: PagedKVStore,
) -> bool:
    """执行一项计划，返回本项是否产生了一个模拟输出 Token。

    部分 Prefill：追加 KV，但不增加输出数量，返回 False。
    历史重算：追加 KV、累计重算量，但不增加输出数量，返回 False。
    Prefill 收尾或 Decode：追加 KV 后增加一个输出，返回 True。
    达到 max_new_tokens：保留输出计数，标记结束并释放请求 KV 引用。

    只承诺本项在输入校验失败或 KV 容量不足时不推进；
    不将 CPU 时间或迭代数解释为真实 TTFT / TPOT。
    """
    # 检查请求 ID、起始位置、阶段和处理数量
    _validate_chunk(request, chunk, store)

    # 为每个待计算位置构造一个绝对位置编号。
    # 例如第二段 Prefill 从 6 开始、处理 4 个位置，得到 [6, 7, 8, 9]。
    positions = torch.arange(
        chunk.start_pos,
        chunk.start_pos + chunk.num_tokens,
        dtype=torch.float32,
        device="cpu",
    )

    # 从 [N] 变成 [N, 1, 1]，再扩展成存储接口需要的 [N, Hkv, D]。
    # 同一位置的各个元素暂时填相同编号，只用于核对位置和写入长度。
    k_new = positions.view(-1, 1, 1).expand(
        chunk.num_tokens,
        store.num_kv_heads,
        store.head_dim,
    ).contiguous()
    v_new = k_new + 1000

    # append 内部负责扩容、写入，以及更新有效 KV 长度。
    # 必须先成功写入，再增加输出数量；容量不足时不能假装已经产生输出。
    store.append(request.kv, k_new, v_new)

    # 必须在普通 Prefill / Decode 的输出反馈之前分流。
    # 否则历史重算一旦超过 prompt_len，就可能被错记为新输出。
    if chunk.phase == "recompute":
        return _finish_recompute(request, chunk)

    # 部分 Prefill 尚未处理到输入末尾，不能产生首个输出。
    if request.kv.num_tokens < request.prompt_len:
        return False

    # 到达这里表示完整 Prefill 收尾，或一次普通 Decode 已执行完毕。
    # 用计数加一模拟采样；这里没有执行 LM head 或真实采样。
    request.num_generated_tokens += 1

    if request.num_generated_tokens >= request.max_new_tokens:
        request.finished = True

        # 最后一个输出无需再送入模型，可以立即释放请求持有的 KV 引用。
        # 若存在共享者或缓存引用，实际物理块仍按原有引用计数规则回收。
        request.kv.release_all()

    return True
