"""教学级请求队列与容量感知调度；不模拟真实 GPU 耗时。

step 从 0 开始，arrival_step 只是离散迭代编号，不是毫秒。
策略：Decode 优先、同类先 running 后 waiting；缺块就跳过当前请求。
允许在迭代边界显式抢占指定请求；不自动选择牺牲者，不淘汰缓存，
也不保证公平性。重算只验证进度和资源账目，不执行真实模型。
"""

from stage_1_paged_kv.paged_kv_store import PagedKVStore
from stage_2_scheduler.scheduler import ScheduledChunk, SchedulingRequest, plan_step
from stage_2_scheduler.scheduler_simulator import execute_chunk


class TeachingScheduler:
    def __init__(
        self,
        store: PagedKVStore,
        max_num_batched_tokens: int,
        max_num_seqs: int,
        max_model_len: int,
    ):
        if min(max_num_batched_tokens, max_num_seqs, max_model_len) <= 0:
            raise ValueError("调度上限必须大于 0")
        self.store = store
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.step_index = 0
        self.requests: dict[str, SchedulingRequest] = {}
        self._future: list[tuple[int, SchedulingRequest]] = []
        self.waiting: list[SchedulingRequest] = []
        self.running: list[SchedulingRequest] = []
        self.finished: list[SchedulingRequest] = []
        self.blocked_requests: list[str] = []
        self._pending_preemptions: list[dict] = []

    def submit(self, request: SchedulingRequest, arrival_step: int | None = None) -> None:
        """登记请求；本步采用保守策略，拒绝输入加输出上限超过总长度的请求。

        默认在当前迭代到达。只做合法性检查，不按 max_model_len
        或整个生成长度预留 KV。
        可接收已有合法 KV 的测试请求，用于构造接近块边界的场景。
        """
        if arrival_step is None:
            arrival_step = self.step_index
        if request.request_id in self.requests:
            raise ValueError("请求 ID 不能重复")
        if arrival_step < self.step_index:
            raise ValueError("不能登记已经过去的到达时刻")
        if request.kv.pool is not self.store.pool or request.kv.block_size != self.store.block_size:
            raise ValueError("请求必须使用调度器对应的块池与块大小")
        if request.prompt_len <= 0 or request.max_new_tokens <= 0:
            raise ValueError("输入长度与输出上限必须大于 0")
        if request.prompt_len + request.max_new_tokens > self.max_model_len:
            raise ValueError("输入长度加输出上限超过 max_model_len")
        if request.finished or not 0 <= request.num_generated_tokens < request.max_new_tokens:
            raise ValueError("只能登记尚未完成且输出计数有效的请求")
        if request.recompute_until != 0:
            raise ValueError("恢复任务应由当前调度器的 preempt 登记，不从外部提交")
        if request.kv.num_tokens < 0 or request.num_pending_tokens <= 0:
            raise ValueError("请求必须有合法的有效 KV 长度和待计算位置")
        if request.is_prefill:
            if request.num_generated_tokens != 0:
                raise ValueError("本步不支持生成后丢失 KV 的恢复重算")
        elif request.num_pending_tokens != 1:
            raise ValueError("普通 Decode 必须恰有一个待计算输出")

        # 所有检查成功之后才登记，拒绝请求时不改变队列或分配任何块。
        self.requests[request.request_id] = request
        self._future.append((arrival_step, request))

    def _collect_arrivals(self) -> None:
        remaining = []
        for arrival_step, request in self._future:
            if arrival_step <= self.step_index:
                self.waiting.append(request)
            else:
                remaining.append((arrival_step, request))
        self._future = remaining

    def preempt(self, request_id: str) -> dict:
        """在迭代边界抢占指定 running 请求，返回释放资源的事件。

        保留输出计数，释放全部请求 KV 引用，移回 waiting。
        只丢弃 KV，不取消请求。共享引用可能使实际释放的块数为 0。
        不支持计划预留后、执行前调用；本教学循环每轮预留后立即执行。
        """
        request = self.requests[request_id]
        if request not in self.running or request.finished:
            raise ValueError("只能抢占当前 running 且尚未结束的请求")
        if request.kv.num_tokens <= 0:
            raise ValueError("请求没有有效 KV 可释放")

        dropped_tokens = request.kv.num_tokens

        # 如果恢复了一部分后再次被抢占，必须保留原来的完整终点。
        # 例如原本要恢复到 8，目前只恢复到 4，终点仍应为 8。
        request.recompute_until = max(
            request.recompute_until,
            dropped_tokens,
        )

        free_before = len(self.store.pool.free_blocks)

        # 只释放 KV 引用，不清空输出计数，也不把请求标记为 finished。
        request.kv.release_all()

        self.running.remove(request)
        self.waiting.append(request)

        # 释放引用不一定释放物理块，因此从块池变化计算实际释放数量。
        # 共享者或缓存仍持有引用时，freed_blocks 可以为 0。
        event = {
            "request_id": request_id,
            "dropped_kv_tokens": dropped_tokens,
            "freed_blocks": (
                len(self.store.pool.free_blocks) - free_before
            ),
        }

        # 保存独立副本，避免调用者修改返回值后污染轨迹记录。
        self._pending_preemptions.append(dict(event))
        return event

    def _reserve_plan(self) -> list[ScheduledChunk]:
        """形成计划并逐项预留容量，但不执行或增加有效 KV 长度。

        成功领取本轮容量后，waiting 请求才转入 running。
        running 仅表示已经接纳，未选中或缺块时仍可留在 running。
        缺块请求记入 blocked_requests，不消耗 Token 预算和序列名额。

        约定：单线程、每轮计划立即执行，无取消；不提供整批事务回滚。
        """
        # 每轮重新记录缺块请求，避免把上一轮原因带入本轮日志。
        self.blocked_requests = []

        candidates = [
            request
            for request in self.running + self.waiting
            if not request.finished and request.num_pending_tokens > 0
        ]

        # False 排在 True 前：Decode 优先于 Prefill。
        # Python 排序是稳定的，因此同一阶段内保留原顺序：
        # 已接纳的 running 请求在前，waiting 请求在后。
        candidates.sort(key=lambda request: request.is_prefill or request.is_recomputing)

        remaining_tokens = self.max_num_batched_tokens
        plan: list[ScheduledChunk] = []

        for request in candidates:
            if remaining_tokens == 0 or len(plan) >= self.max_num_seqs:
                break

            # 复用原有数量计算逻辑，只为当前这一条请求生成候选指令。
            # candidates 已筛掉没有待计算位置的请求，因此这里必然有一项。
            chunk = plan_step([request], remaining_tokens, 1)[0]

            # 只预留本轮执行到的位置，不预留整个未来输出长度。
            target_tokens = request.kv.num_tokens + chunk.num_tokens

            try:
                request.kv.ensure_capacity(target_tokens)
            except RuntimeError:
                # ensure_capacity 在缺块时保持原状态不变。
                # 本项没有成功安排，不扣预算、不占名额，继续检查其他请求。
                self.blocked_requests.append(request.request_id)
                continue

            # 成功领取本轮容量之后，才把等待请求接纳为 running。
            # 原本就在 running 的请求不重复加入。
            if request in self.waiting:
                self.waiting.remove(request)
                self.running.append(request)

            plan.append(chunk)
            remaining_tokens -= chunk.num_tokens

        return plan

    def step(self) -> dict:
        """推进一轮并返回快照；即使计划为空也推进迭代编号，不忙等。"""
        self._collect_arrivals()
        plan = self._reserve_plan()
        emitted = []
        for chunk in plan:
            request = self.requests[chunk.request_id]
            if execute_chunk(request, chunk, self.store):
                emitted.append(request.request_id)

        # 用新列表收集幸存者，避免遍历时删除 running 中的成员。
        survivors = []
        for request in self.running:
            if request.finished:
                self.finished.append(request)
            else:
                survivors.append(request)
        self.running = survivors

        # 这是整轮执行后的快照；used_blocks 不表示本轮峰值占用。
        states = {request.request_id: "not_arrived" for _, request in self._future}
        for name, queue in [("waiting", self.waiting), ("running", self.running),
                            ("finished", self.finished)]:
            states.update({request.request_id: name for request in queue})
        trace = {
            "step": self.step_index,
            "plan": tuple(plan),
            "emitted": tuple(emitted),
            "blocked": tuple(self.blocked_requests),
            "preemptions": tuple(dict(event) for event in self._pending_preemptions),
            "recomputed_tokens": sum(chunk.num_tokens for chunk in plan
                                     if chunk.phase == "recompute"),
            "total_recomputed_tokens": {name: request.num_recomputed_tokens
                                        for name, request in self.requests.items()},
            "used_tokens": sum(chunk.num_tokens for chunk in plan),
            "used_blocks": self.store.pool.num_blocks - len(self.store.pool.free_blocks),
            "states": states,
            "kv_tokens": {name: request.kv.num_tokens for name, request in self.requests.items()},
            "block_tables": {name: tuple(request.kv.block_table)
                             for name, request in self.requests.items()},
        }
        self._pending_preemptions = []
        self.step_index += 1
        return trace
