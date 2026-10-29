# Prefill/Decode 调度与抢占
from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs                         # 一轮最多几条序列
        self.max_num_batched_tokens = config.max_num_batched_tokens     # 一轮最多几个 token
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()     # 新来的、还没 prefill 的序列
        self.running: deque[Sequence] = deque()     # 已 prefill、正在逐 token decode 的序列

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = [] # 收集本轮最终被调度到 GPU 的序列
        num_batched_tokens = 0 # 累计本轮已经计算完毕的 token 总数

        # prefill 优先原则
        # 检查 waiting 队列
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            # 计算剩余 token 预算
            remaining = self.max_num_batched_tokens - num_batched_tokens    
            if remaining == 0:
                break

            # 显存分配检查：首次调度，先探测前缀缓存并占块
            if not seq.block_table:     # seq 刚来的时候，block_table 是空的
                # 调度器通过 BlockManager 检查 GPU 显存池里有没有空闲的物理块
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:     # 块不够，放不下
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:       # chunked prefill 续算：块已分配
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            if remaining < num_tokens and scheduled_seqs:  # 仅允许对第一个序列的 prefill 分块
                break       # 预算不够、且已有序列：收手

            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            # 允许对「本轮第一条」做 chunked prefill：只算预算内的部分
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens

            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING     # 整段 prompt 算完才转 running
                self.waiting.popleft()      # 移出 waiting 队列
                self.running.append(seq)    # 移入 running 队列

            # 没算完，加入调度列表，此时仍处于 waiting
            scheduled_seqs.append(seq)

        # 跳出循环后，检查当前调度队列是否为空，不为空，继续调度之前没算完的序列
        # 这意味着只要有 prefill 的序列，直接 return，不进入 decode。这是和官方 vllm 的差异
        if scheduled_seqs:
            return scheduled_seqs, True     # 第二个布尔值 True 代表本轮是 Prefill 阶段

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())        # 牺牲队尾序列（LIFO）
                else:
                    self.preempt(seq)                       # 没得牺牲了，牺牲自己
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False                  # 已过 prefill，进入逐 token 生成
                self.block_manager.may_append(seq)      # 真的跨块就取新块
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))       # 放回队头，保持顺序
        return scheduled_seqs, False                            # is_prefill = False

    # 抢占 
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING     # 打回等待队列
        seq.is_prefill = True                   # 下次要重新 prefill
        self.block_manager.deallocate(seq)      # 释放它占的所有 KV 块
        self.waiting.appendleft(seq)            # 插回队头，优先重新调度


    # 把结果写回序列并判断是否结束
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            # 把本轮算过的 token 记为「已缓存」（chunked prefill 可能分多步）
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue                        # prefill 还没算完，先不追加新 token

            seq.append_token(token_id)          # 追加采样出的新 token
            
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)      # 结束就释放 KV
                self.running.remove(seq)
