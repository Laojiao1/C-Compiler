"""前缀缓存的内容标识：先建立完整块的哈希链，再逐步接入物理块。"""

import hashlib
import json
from collections import OrderedDict

from paged_kv import BlockPool, RequestKV


def build_prefix_hashes(
    token_ids: list[int], block_size: int, namespace: str
) -> list[str]:
    """返回从输入开头开始、每个完整逻辑块对应的前缀指纹。

    token_ids：从位置 0 开始的 Token ID 序列，本步使用人工构造的整数。
    block_size：每块容纳的位置数；未满的最后一块不参与缓存标识。
    namespace：区分模型版本、KV dtype 等计算条件，由调用者固定并传入。

    输出第 i 项对应的是“前 i+1 个完整块”的内容上下文，
    而不是只由第 i 块内部的 Token 决定。函数不修改输入列表。

    这里只生成内容指纹，不访问块池、不保留 K/V、不进行物理块复用。
    """
    if block_size <= 0:
        raise ValueError("block_size 必须大于 0")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace 必须是非空字符串")
    if any(type(token) is not int or token < 0 for token in token_ids):
        raise ValueError("Token ID 必须为非负整数")

    # 遍历完整块，将计算条件、父哈希和本块 Token 一起序列化，
    # 使用 SHA-256 得到本块指纹，再把它作为下一块的父哈希。
    hashes = []

    full_tokens = len(token_ids) // block_size * block_size
    parent_hash = ""

    for start in range(0, full_tokens, block_size):
        # 取出当前完整块。例如 start=4 时，取得位置 4、5、6、7。
        block_tokens = token_ids[start: start + block_size]
        # 指纹同时描述“怎么算”和“算什么”：
        # namespace 区分计算条件；block_size 明确块划分方式；
        # parent_hash 把前面整个上下文连接进来；
        # block_tokens 则描述当前这一块的输入。
        payload = {
            "namespace": namespace,
            "block_size": block_size,
            "parent_hash": parent_hash,
            "token_ids": block_tokens,
        }

        # SHA-256 接收字节。先将结构转换成稳定的 JSON，再编码成字节。
        # 固定字段顺序与分隔符，使相同内容在不同进程中得到相同输入。
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False
        ).encode("utf-8")

        # 得到固定长度的内容指纹。这里使用 SHA-256，避免 Python 内置 hash() 的进程随机化影响后续跨进程比较
        block_hash = hashlib.sha256(encoded).hexdigest()
        hashes.append(block_hash)

        # 下一块必须携带当前块的指纹，形成前缀哈希链
        parent_hash = block_hash

    return hashes


class PrefixCache:
    """同一块池和计算条件下的完整块缓存，使用单线程教学实现。

    entries 把前缀指纹映射到物理块号，每条记录持有一个块引用。
    因而本模型的 ref_counts = 请求引用数 + 缓存条目引用数。
    这与 vLLM 中“活跃请求引用为 0，仍可保留可淘汰内容”的计数口径不同。

    OrderedDict 的前端表示较久未使用，命中后移到末尾，供后续淘汰使用。
    本实验每个块池只使用一个缓存实例，淘汰通过 evict_one 显式触发；
    容量不足时是否淘汰、淘汰多少块，仍由调用者决定。
    """

    def __init__(self, pool: BlockPool, block_size: int, namespace: str):
        if block_size <= 0:
            raise ValueError("block_size 必须大于 0")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("namespace 必须是非空字符串")
        self.pool = pool
        self.block_size = block_size
        self.namespace = namespace
        self.entries: OrderedDict[str, int] = OrderedDict()

    def _registration_hashes(
        self, request: RequestKV, token_ids: list[int]
    ) -> list[str]:
        """先检查所有输入，避免登记到一半才发现请求不匹配。"""
        if request.pool is not self.pool or request.block_size != self.block_size:
            raise ValueError("请求必须与缓存使用相同块池和 block_size")
        if len(token_ids) != request.num_tokens:
            raise ValueError("登记的 Token IDs 必须恰好对应已写入的有效历史")
        # Token 与 K/V、namespace 的语义对应由调用者保证；这里不运行模型验证。
        return build_prefix_hashes(token_ids, self.block_size, self.namespace)

    def register(self, request: RequestKV, token_ids: list[int]) -> None:
        """登记已经计算完成的完整块；重复指纹沿用已有条目，不重复持有引用。"""
        hashes = self._registration_hashes(request, token_ids)
        for block_hash, block_id in zip(hashes, request.block_table):
            # 同样的前缀可能被重复登记，甚至已经在另一组物理块上重复算过。
            # 沿用已有缓存条目，避免重复增加引用、占住更多存储。
            if block_hash in self.entries:
                continue

            # 为“缓存本身”增加一个引用。
            # 之后即使原请求结束，这个块也不会降为 0、进入空闲队列。
            self.pool.retain(block_id)

            # 保存指纹到物理块的映射。这里没有复制任何 K/V 张量。
            self.entries[block_hash] = block_id
        

    def acquire(self, token_ids: list[int]) -> RequestKV:
        """从开头查找连续命中的完整块，返回已持有命中块引用的新请求。

        返回请求的 num_tokens 是命中的位置数，其余输入尚未计算。
        第一次未命中时停止；不能跨过缺口领取后面的块。
        命中条目移到 entries 末尾，表示最近使用过。
        """
        # 新请求拥有独立块表，但可以引用缓存中已有的物理块。
        request = RequestKV(self.pool, self.block_size)

        hashes = build_prefix_hashes(token_ids, self.block_size, self.namespace)
        for block_hash in hashes:
            if block_hash not in self.entries:
                break
            block_id = self.entries[block_hash]
            self.pool.retain(block_id)
            request.block_table.append(block_id)
            self.entries.move_to_end(block_hash)

        request.num_tokens = len(request.block_table) * self.block_size

        return request

    def evict_one(self) -> int | None:
        """淘汰较久未使用、且仅剩缓存引用的一块，返回物理块号。

        按 entries 从前往后查找，跳过仍被请求使用的块。
        ref_counts == 1 表示只有当前缓存条目持有引用，可以真正归还空间。
        缓存为空或所有条目仍被请求引用时返回 None，且不改变状态。

        只移除缓存映射并释放缓存引用，不清零物理 K/V 张量。
        """
        victim = None
        for block_hash, block_id in self.entries.items():
            if self.pool.ref_counts[block_id] == 1:
                victim = (block_hash, block_id)
                break

        if victim is None:
            return None

        block_hash, block_id = victim

        # 先删除 hash 再释放引用，否则当物理块被重新分配、写入新数据后，旧指纹仍可能把其他请求错误地引向这块新数据
        del self.entries[block_hash]
        self.pool.release(block_id)

        return block_id

        
