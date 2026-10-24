"""5.3.1 教学级分页 KV Cache：从地址映射开始，逐步扩展。"""

from collections import deque


def locate_token(
    block_table: list[int], block_size: int, position: int
) -> tuple[int, int]:
    """把逻辑位置转换为 (物理块号, 块内偏移)。"""
    if block_size <= 0:
        raise ValueError("block_size 必须 > 0")

    # Python 允许负下标，这里必须显式拒绝负位置。
    if position < 0:
        raise ValueError("position 不能小于 0")

    logical_block = position // block_size
    offset = position % block_size

    if logical_block >= len(block_table):
        raise IndexError("该逻辑位置尚未分配物理块")

    physicial_block = block_table[logical_block]

    return physicial_block, offset


class BlockPool:
    """管理物理块号；本步还不存储实际 K/V 张量。"""

    def __init__(self, num_blocks: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks 必须大于 0")

        self.num_blocks = num_blocks
        self.free_blocks: deque[int] = deque(range(num_blocks))
        # 0 表示可归还到空闲队列；请求和缓存条目都可以持有引用。
        # 教学模型采用总持有者计数，与 vLLM 的活跃请求引用计数口径不同。
        self.ref_counts: list[int] = [0] * num_blocks

    def allocate(self) -> int:
        """领取一个物理块号；池耗尽时抛出 RuntimeError，保持状态不变。"""
        if not self.free_blocks:
            raise RuntimeError("没有可分配的物理块")

        block_id = self.free_blocks.popleft()
        self.ref_counts[block_id] = 1
        return block_id

    def retain(self, block_id: int) -> None:
        """为已分配块增加一个引用，不改变空闲队列。

        块号越界：IndexError；引用空闲块：ValueError。
        空闲块必须通过 allocate 领取，不能用 retain 跳过分配。
        """
        if block_id < 0 or block_id >= self.num_blocks:
            raise IndexError("物理块号越界")

        if self.ref_counts[block_id] == 0:
            raise ValueError("不能引用空闲块")

        self.ref_counts[block_id] += 1

    def release(self, block_id: int) -> None:
        """释放一个引用；降到 0 时将块号归还队列尾部。

        块号越界：IndexError；释放空闲块：ValueError。
        """
        if block_id < 0 or block_id >= self.num_blocks:
            raise IndexError("物理块号越界")

        if self.ref_counts[block_id] == 0:
            raise ValueError("不能释放空闲块")

        self.ref_counts[block_id] -= 1

        if self.ref_counts[block_id] == 0:
            self.free_blocks.append(block_id)


class RequestKV:
    """维护单条请求的块表；容量预留与实际 K/V 写入分开。"""

    def __init__(self, pool: BlockPool, block_size: int):
        if block_size <= 0:
            raise ValueError("block_size 必须大于 0")
        self.pool = pool
        self.block_size = block_size
        self.block_table: list[int] = []
        # 预留槽位不增加有效长度；只有 K/V 成功写入后才增加。
        self.num_tokens = 0

    def ensure_capacity(self, target_tokens: int) -> None:
        if target_tokens < 0:
            raise ValueError("target_tokens 不能小于 0")

        required_blocks = (target_tokens + self.block_size - 1) // self.block_size
        missing_blocks = max(0, required_blocks - len(self.block_table))

        if missing_blocks > len(self.pool.free_blocks):
            raise RuntimeError("空闲物理块不足，无法满足请求容量")

        for _ in range(missing_blocks):
            block_id = self.pool.allocate()
            self.block_table.append(block_id)

    def share_prefix(self, prefix_tokens: int) -> "RequestKV":
        """返回共享指定完整前缀的新请求，不分配新物理块、不复制 K/V。

        prefix_tokens 必须是 block_size 的整数倍，且不超过有效长度。
        本步由调用者确认共享前缀正确，尚不通过 Token IDs/hash 自动匹配。
        """
        if prefix_tokens < 0 or prefix_tokens > self.num_tokens:
            raise ValueError("共享前缀长度必须位于已写入的有效历史范围内")
        if prefix_tokens % self.block_size != 0:
            raise ValueError("本步只共享完整块，前缀长度必须按块对齐")

        shared = RequestKV(self.pool, self.block_size)
        num_shared_blocks = prefix_tokens // self.block_size

        for block_id in self.block_table[:num_shared_blocks]:
            self.pool.retain(block_id)
            shared.block_table.append(block_id)

        shared.num_tokens = prefix_tokens
        return shared

    def release_all(self) -> None:
        """归还本请求的全部块引用并清空块表；允许重复调用。"""
        for block_id in self.block_table:
            self.pool.release(block_id)

        # 清除归属记录，避免下次再次释放同一批引用。
        self.block_table.clear()
        self.num_tokens = 0
