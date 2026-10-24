"""单层 CPU/FP32 KV 存储 reference；用来学习地址与数据的对应关系。"""

import torch

from stage_1_paged_kv.paged_kv import BlockPool, RequestKV, locate_token


class PagedKVStore:
    def __init__(
        self, pool: BlockPool, block_size: int, num_kv_heads: int, head_dim: int
    ):
        if min(block_size, num_kv_heads, head_dim) <= 0:
            raise ValueError("block_size、num_kv_heads、head_dim 必须大于 0")
        self.pool = pool
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        shape = (pool.num_blocks, block_size, num_kv_heads, head_dim)
        self.k_pool = torch.empty(shape, dtype=torch.float32, device="cpu")
        self.v_pool = torch.empty(shape, dtype=torch.float32, device="cpu")

    def _check_request(self, request: RequestKV) -> None:
        if request.pool is not self.pool:
            raise ValueError("请求与存储必须使用同一个块池")
        if request.block_size != self.block_size:
            raise ValueError("请求与存储的 block_size 必须相同")

    def _check_append_inputs(
        self, request: RequestKV, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> None:
        self._check_request(request)
        expected_tail = (self.num_kv_heads, self.head_dim)
        if (
            k_new.ndim != 3
            or v_new.shape != k_new.shape
            or tuple(k_new.shape[1:]) != expected_tail
        ):
            raise ValueError("K/V 必须同为 [N, num_kv_heads, head_dim]")
        for tensor in (k_new, v_new):
            if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
                raise ValueError("本步只接收 CPU 上的 float32 K/V")

    @torch.no_grad()
    def append(
        self, request: RequestKV, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> None:
        """追加 N 个位置的 K/V；成功后才增加 request.num_tokens。

        先检查输入和容量，再写入。容量不足时请求与块池状态不变。
        共享仅限已写满的前缀块，追加位置始终位于共享前缀之后。
        """
        self._check_append_inputs(request, k_new, v_new)

        old_tokens = request.num_tokens
        new_tokens = old_tokens + k_new.shape[0] 

        request.ensure_capacity(new_tokens)

        for i in range(k_new.shape[0]):
            position = old_tokens + i
            block_id, offset = locate_token(request.block_table, self.block_size, position)

            self.k_pool[block_id, offset].copy_(k_new[i])
            self.v_pool[block_id, offset].copy_(v_new[i])

        request.num_tokens = new_tokens

    def gather(self, request: RequestKV) -> tuple[torch.Tensor, torch.Tensor]:
        """返回独立的连续 K/V，shape 均为 [num_tokens, Hkv, D]。

        只读取有效位置，既不返回尾块空槽，也不读取额外预留块。
        该方法为了数值验证而复制数据，不模拟高效分页 kernel 的执行方式。
        """
        self._check_request(request)
        if request.num_tokens == 0:
            shape = (0, self.num_kv_heads, self.head_dim)
            return self.k_pool.new_empty(shape), self.v_pool.new_empty(shape)
        
        k_parts = []
        v_parts = []

        for start in range(0, request.num_tokens, self.block_size):
            logical_block = start // self.block_size
            block_id = request.block_table[logical_block]

            # 最后一块可能没满，只取已经写入的位置。
            count = min(
                self.block_size, request.num_tokens - start
            )

            k_parts.append(self.k_pool[block_id, :count])
            v_parts.append(self.v_pool[block_id, :count])

        return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)
