# KV Cache 与请求调度教学实践

当前使用单层、CPU/FP32 小张量，验证块管理、K/V 读写、Attention 数值结果与完整块前缀共享。
学习讲义与阶段性实验记录存放于既有阶段五笔记目录。

- `paged_kv.py`：地址转换、物理块池、请求块表与有效长度。
- `paged_kv_store.py`：物理 K/V 张量及追加、收集操作。
- `attention_reference.py`：单 Query Attention 核心。
- `prefix_cache.py`：完整前缀块指纹，以及指纹到物理块的缓存索引。
- `tests/`：全部单元测试及独立的 PyTorch SDPA 对照。

使用已有 WSL `vllm` 环境，在本目录统一执行：

```bash
python -m unittest discover -s tests -v
```

只运行某一组，例如 K/V 读写：

```bash
python -m unittest -v tests.test_kv_store
```

Attention reference 使用 `gather()` 后的连续历史进行计算，用于数值验证；
不提供 GPU 分页 kernel 或性能结论。

已完成通过 `RequestKV.share_prefix()` 显式共享完整块。
已完成 `build_prefix_hashes()`，根据 Token IDs 和计算条件生成哈希链。
已完成 `PrefixCache.register()` 与 `acquire()`，关联和领取已计算的完整块。
缓存条目持有独立引用，使源请求结束后内容仍有效；这改变了本教学模型的计数口径：
`ref_counts = 请求引用数 + 缓存条目引用数`，不是 vLLM 的活跃请求引用数。
已完成 `PrefixCache.evict_one()`，显式淘汰较久未使用且没有请求引用的缓存块。
本实验每个块池使用一个缓存实例；仅剩缓存引用的块满足 `ref_counts == 1`。
淘汰释放一块后，由调用者决定是否继续淘汰及重试分配，尚无自动容量管理。

2026-10-21 收到的用户验收输出为 51 项测试通过，见
[原始日志](results/raw/tests_pass_20261021.log)。
[归档时源码指纹](results/source_fingerprints_20261021.json)用于后续状态对照，
不是对用户执行测试时源码版本的独立快照。
