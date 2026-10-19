# GPT-2 显存与计算模型

本文记录阶段 5.1.4 的容量估算、CUDA 实测和解释边界。显存单位统一使用 MiB：`1 MiB = 1024 × 1024 bytes`。

## 实验条件

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| PyTorch | 2.11.0+cu128 |
| 模型 | GPT-2 Small，124,439,808 个唯一参数 |
| batch size | 1 和 2 |
| 序列长度 | 128、512、1024 |
| dtype | FP32、FP16 |
| forward 模式 | `torch.inference_mode()`，eval |

测量脚本先对每个长度做一次 warmup，然后在 forward 前记录 baseline，随后读取 `max_memory_allocated`、forward 结束时仍分配的显存和 `max_memory_reserved`。Cache 测量使用一块与所有层 K/V 元素总量相同的 CUDA tensor，比较分配前后的 `memory_allocated`。

## 权重容量

GPT-2 Small 有 124,439,808 个唯一参数。Token Embedding 和 LM Head 共享参数，因此只计一次。

```text
FP32: 124,439,808 × 4 bytes = 497,759,232 bytes = 474.70 MiB
FP16: 124,439,808 × 2 bytes = 248,879,616 bytes = 237.35 MiB
```

模型还注册了每层一张固定 causal mask。每张是 `1024 × 1024` 个 bool，12 层共约 12 MiB。实测 unique registered tensor storage 为 FP32 486.70 MiB、FP16 249.35 MiB，与“参数 + buffers”相符。

## KV Cache 容量

GPT-2 Small 使用标准 MHA，单请求 Cache 容量为：

```text
2 × num_layers × batch × sequence_length × hidden_size × dtype_bytes
```

最前面的 2 分别表示 K 和 V。由于 `12 heads × 64 head_dim = hidden_size 768`，也可以将公式写成 `2 × layers × batch × sequence_length × heads × head_dim × dtype_bytes`。

### 理论值与 CUDA 分配增量

| dtype | batch | 序列长度 | KV 理论容量 | `memory_allocated` 增量 |
|---|---:|---:|---:|---:|
| FP32 | 1 | 128 | 9 MiB | 9 MiB |
| FP32 | 1 | 512 | 36 MiB | 36 MiB |
| FP32 | 1 | 1024 | 72 MiB | 72 MiB |
| FP16 | 1 | 128 | 4.5 MiB | 4.5 MiB |
| FP16 | 1 | 512 | 18 MiB | 18 MiB |
| FP16 | 1 | 1024 | 36 MiB | 36 MiB |
| FP16 | 2 | 128 | 9 MiB | 9 MiB |
| FP16 | 2 | 512 | 36 MiB | 36 MiB |
| FP16 | 2 | 1024 | 72 MiB | 72 MiB |

实测增量与理论值逐项吻合。Cache 长度、batch 或 dtype 字节数翻倍时，理论容量按同一比例增长。GQA/MQA 模型需要把公式中的 head 数换成 `num_kv_heads`。

## Forward 输出与峰值额外显存

forward logits shape 是 `[batch, sequence_length, 50257]`。仅 logits 的理论容量为：

```text
batch × sequence_length × vocab_size × dtype_bytes
```

例如 FP32、batch=1、长度=1024 时，logits 占约 196.32 MiB。forward 结束时的 `memory_allocated - baseline` 与 logits 大小大致相符；它表示 forward 返回后仍活着的分配，并不必然只包含 logits。

### 实测值

下表中的 `forward peak extra` 是 `max_memory_allocated - forward 前 baseline`。`live after forward` 是 forward 返回后仍分配的增量。两者来自不同时间点，不能相减来计算 workspace。

| dtype | batch | 长度 | forward peak extra | live after forward | reserved peak (baseline) |
|---|---:|---:|---:|---:|---:|
| FP32 | 1 | 128 | 29.79 MiB | 24.54 MiB | 576 (576) MiB |
| FP32 | 1 | 512 | 126.16 MiB | 98.16 MiB | 676 (676) MiB |
| FP32 | 1 | 1024 | 238.32 MiB | 196.32 MiB | 910 (910) MiB |
| FP16 | 1 | 128 | 14.90 MiB | 12.27 MiB | 288 (288) MiB |
| FP16 | 1 | 512 | 60.50 MiB | 50.00 MiB | 368 (368) MiB |
| FP16 | 1 | 1024 | 119.67 MiB | 98.16 MiB | 468 (468) MiB |
| FP16 | 2 | 128 | 29.79 MiB | 24.54 MiB | 322 (322) MiB |
| FP16 | 2 | 512 | 119.29 MiB | 98.16 MiB | 434 (434) MiB |
| FP16 | 2 | 1024 | 238.45 MiB | 196.32 MiB | 668 (668) MiB |

标准 eager Attention 的 score shape 为 `[B, 12, T, T]`。例如 FP32、batch=1、长度=1024 时，单份 score 张量理论上占 48 MiB。softmax 和其他中间结果也会占用临时空间；但这些张量的生命周期不同，最终峰值不能简单地把所有 shape 的容量相加。

## allocated、reserved 和 GPU 总占用

- `memory_allocated` 表示 PyTorch 当前有活跃 tensor 占用的显存。
- `max_memory_allocated` 是测量区间内活跃 tensor 占用的最高值。
- `memory_reserved` 是 PyTorch 向 CUDA 申请并由 caching allocator 保留、可供后续复用的空间；其中有些块当时不属于活跃 tensor。
- `nvidia-smi` 还包含 CUDA context、其他进程和驱动侧开销，因此数值通常不等于 PyTorch 的 allocated 或 reserved。

本次每个长度的 reserved peak 与 baseline 相同，说明 forward 使用了分配器此前已经保留的空间，没有进一步扩大 reserved pool。它不代表 forward 没有临时 tensor；活跃峰值由 `max_memory_allocated` 反映。

## 目前未解释的模型加载差额

参数与注册 buffer 的 unique storage 是 FP32 486.70 MiB、FP16 249.35 MiB。模型加载后 PyTorch 的 allocated 读数分别为 FP32 487.46 MiB、FP16 batch=1 267.23 MiB、FP16 batch=2 255.48 MiB。差额分别是 0.76、17.88 和 6.13 MiB。

FP16 两次模型加载读数的差额不同，因此目前没有证据把它说成固定的模型权重、因果 mask 或确定的分配器开销。后续若需要精确归因，应进一步记录 CUDA active allocation snapshot 和分配调用栈。本报告把这部分保留为未归属的活跃 CUDA 分配，不将其混入理论参数或 KV Cache 容量。

FP16、batch=1、长度=512 时，forward 后 live 增量为 50 MiB，而 logits 理论容量约 49.08 MiB。额外约 0.92 MiB 也说明该指标可能包含其他仍活着的小 tensor，不能直接将整项标记为 logits。

## Decode 的简化计算与访存估算

算术强度是“完成的浮点运算数 / 从显存读取或写入的字节数”。它帮助判断某类工作更容易受计算吞吐还是数据搬运速度限制。

单 token Decode 的 Attention 对历史长度 T 做两次主要矩阵运算：QKᵀ 和概率乘 V。忽略 softmax 等操作，计算量约为 `4 × B × T × hidden_size` FLOPs；读取 K 和 V 的字节数约为 `2 × B × T × hidden_size × dtype_bytes`。因此这个 Attention 部分的粗略算术强度约为 `2 / dtype_bytes` FLOP/byte：FP16 约 1 FLOP/byte，FP32 约 0.5 FLOP/byte。

这个估算只看历史 K/V 读取和两个矩阵乘法，没有包含 QKV/MLP 权重读取、softmax、Cache 写入、并行度和 kernel launch。它提示单 token Attention 可能受到带宽压力，但不能单独判定完整 Decode 是带宽瓶颈。5.1.5 会通过 workload sweep 和 profiler 检验实际瓶颈。

## 结论与边界

- 权重和 KV Cache 的理论容量与实测关系清楚；KV Cache 实测分配增量完全吻合公式。
- Cache 对 batch、长度和 dtype 的线性关系已通过 batch=1/2、FP32/FP16、多个长度验证。
- Prefill forward 峰值显著高于模型基线；它包含 logits 及计算过程中的临时张量，不能把峰值都归因于 Attention score。
- reserved 是可复用内存池的容量，不能与 allocated 相加当成 tensor 使用量。
- FP16 模型加载时存在变化的未归属 allocated 差额，已明确保留为待追踪项。
- 简化算术强度说明 Decode Attention 的 K/V 读取值得关注，但真正瓶颈需要 profiler 证据。
