# 5.1.5 GPT-2 Benchmark 与 Profiler 报告

本报告记录 RTX 5060 Laptop GPU 上 GPT-2 Small 的 Prefill/Decode workload sweep，以及 Nsight Systems 对一组代表性 shape 的 CUDA trace。显存使用 MiB（`1 MiB = 1024 × 1024 bytes`），延迟使用毫秒。

## 实验环境与边界

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，compute capability `(12, 0)` |
| PyTorch | `2.11.0+cu128` |
| CUDA runtime | `12.8` |
| 模型 | 手写 GPT-2 Small，Hugging Face `gpt2` 权重 |
| dtype / Cache | FP16，预分配 KV Cache |
| benchmark warmup / measured | 每组 3 / 10 次 |

计时从输入已在 GPU 后开始，并在计时边界同步 CUDA。Prefill 计时包含 prompt embedding、各层前向、建 Cache 和整段 prompt logits；TTFT 是 Prefill 加首 token argmax。每个后续 Decode step 包含单 token 前向和下一个 token 的 argmax。端到端 tokens/s 用整个 batch 的输出 token 总数除以 TTFT 与后续 Decode 时间之和。

扫描时序列长度为 32 和 128，输出长度为 8 和 32。脚本每组 shape 单独 warmup，再执行 10 个 measured run；TPOT p50 是各轮后续 Decode 总时间的中位数除以 Decode 步数，step p95 则来自各轮逐步延迟样本。

## Benchmark 结果

| Batch | Prompt | Output | Prefill p50 | TTFT p50 | TPOT p50 | Decode step p95 | 总 tokens/s | Peak extra | Peak reserved |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 8 | 7.532 | 7.598 | 5.552 | 7.414 | 172.18 | 4.73 MiB | 290 MiB |
| 1 | 32 | 32 | 4.256 | 4.316 | 5.208 | 19.797 | 193.03 | 5.57 MiB | 292 MiB |
| 1 | 128 | 8 | 4.967 | 5.033 | 3.845 | 15.212 | 250.39 | 17.25 MiB | 312 MiB |
| 1 | 128 | 32 | 4.764 | 4.825 | 3.423 | 6.159 | 288.41 | 18.09 MiB | 314 MiB |
| 2 | 32 | 8 | 4.895 | 4.958 | 3.535 | 17.748 | 538.65 | 9.34 MiB | 314 MiB |
| 2 | 32 | 32 | 5.182 | 5.245 | 3.983 | 7.143 | 497.17 | 11.02 MiB | 314 MiB |
| 2 | 128 | 8 | 5.653 | 5.727 | 3.985 | 7.481 | 475.91 | 34.49 MiB | 342 MiB |
| 2 | 128 | 32 | 9.097 | 9.170 | 4.929 | 7.578 | 395.15 | 36.18 MiB | 342 MiB |

这里的 `peak extra` 是 `max_memory_allocated - model/input baseline`，包含 Cache、logits 和 forward 临时张量，不是单独的 KV Cache。`peak reserved` 是 allocator 的保留池峰值，不应与 allocated 相加。

## 结果怎么读

### 显存关系较稳定

在同一 prompt 下，batch 从 1 增到 2，peak extra 大致翻倍；例如 prompt=128、output=32 时从 18.09 MiB 到 36.18 MiB。output 从 8 增至 32 时，预分配 Cache 容量变大，因此 peak extra 也略有增加。这与权重之外的 logits、Cache 和临时张量会随 batch、序列长度变化的预期相符。

### 延迟数据存在明显抖动

Prefill 理论上主要由 prompt shape 决定，不应因为要求多生成几个 token 就显著变化。但 B=1、prompt=32 的 output=8 结果为 7.532 ms，output=32 却为 4.256 ms；B=1、prompt=128 也有类似的非单调现象。Decode TPOT 和 p95 同样出现较大差异。短时间、小 batch 的 GPU 工作容易受到 GPU 时钟状态、系统调度以及 Python launch 过程的影响，当前 10 轮样本不足以把这些反常差异归因到某个算子。

因此这组数据适合作为本机、当前协议下的局部基线和显存趋势证据，不适合声称 prompt 更长反而更快，或某个 output 长度本身造成了确定的加速。

## Nsight Systems 证据

### 采集到的设备活动

文件 `profiles/gpt2_b1_p128_o32.nsys-rep` 是 Nsight Systems 原始报告，`profiles/gpt2_b1_p128_o32.sqlite` 是其导出的活动数据库。SQLite 中存在 CUDA kernel、runtime API、memcpy 和 synchronization 活动。`profiles/gpt2_b1_p128_o32.json` 是此前 PyTorch Profiler 导出的 trace；该次 CUPTI 初始化失败，不能作为 GPU kernel 时间证据。

PyTorch Profiler 独立诊断中，`torch.profiler.supported_activities()` 虽然列出了 CUDA，但实际初始化报 `CUPTI_ERROR_INVALID_DEVICE`，最终只出现 CPU 时间列。Nsight Systems 的数据库则记录了 CUDA 活动，因此后续 GPU 分析依据 `.sqlite` / `.nsys-rep`。

### CUDA kernel 统计

Nsight SQLite 汇总得到：

| 指标 | 观察值 |
|---|---:|
| CUDA kernel 记录数 | 18,907 |
| kernel 执行时间求和 | 83.129 ms |
| 整段 trace 时间跨度 | 635.712 ms |
| CUDA runtime `cudaLaunchKernel` 调用 | 18,761 次 |
| `cudaDeviceSynchronize` 调用 | 133 次，API 时间求和 18.545 ms |
| `cudaStreamSynchronize` 调用 | 160 次，API 时间求和 8.425 ms |

最长的四类 CUDA kernel 都是 cuBLAS GEMV 变体，分别记录了 15.854、15.577、13.582 和 12.148 ms；合计 57.161 ms，约占本次 83.129 ms kernel 时间的 68.8%。LayerNorm kernel 时间求和约 3.786 ms；softmax warp kernel 约 0.948 ms；argmax reduction 约 0.549 ms，共 64 次调用。

这说明当前 eager 路径中，许多窄矩阵/向量运算及其 kernel 启动占据了可观的设备时间。Decode 每层包含小型投影和注意力计算，batch=1 时这些运算形状较窄，cuBLAS 常以 GEMV 形式执行，是合理的重点观察对象。64 次 argmax 调用也与两次完整执行（warmup 与 measured）中的首 token 加后续 token 采样次数相符。

### 把启动与设备执行分开

CUDA runtime 中 `cudaLaunchKernel` 的 API 时间求和为 305.199 ms，133 次 `cudaDeviceSynchronize` 的 API 时间求和为 18.545 ms。这些是 host 侧 runtime 调用所花的时间；API 执行与 GPU kernel 可以交叠，不能把 305.199 ms 加到 83.129 ms 上，或把它解释成 GPU kernel 执行时间。

同步调用数也受到测量方法影响：benchmark 为了分别测每个 Prefill、采样和 Decode step，在边界处显式调用同步。Nsight 捕获了 warmup 与 measured 两次执行，所以 trace 中包含大量用于计时的同步。实际服务会在需要返回 token、检查停止条件或交付结果时发生同步，但不一定在每一步都使用相同的主机等待策略。因此这份 trace 能揭示同步和 launch 的成本来源，不能直接等同于生产服务的无插桩时间。

## 本次瓶颈判断与限制

当前最可靠的观察是：CUDA 设备时间主要落在几类 cuBLAS GEMV kernel 上，多个小型算子与大量 launch/synchronization 也值得关注。它支持“窄形状的矩阵运算和 eager 调度开销可能是当前 decode 路径的重要成本”这一判断，但还不足以将某一种 GEMV 精确对应到某一层的 QK、AV、MLP 或 LM Head。

Nsight 捕获了 Python 进程启动、模型权重加载、warmup 和 measured 调用；脚本也没有为 Prefill 与 Decode 添加 NVTX 阶段范围。因此 83.129 ms 是整段捕获中的 kernel 时间总和，不是一次 Decode 的 kernel 时间。kernel 名称可以分类，调用数可以和模型结构交叉核对，但若要严格分出 Prefill/Decode 的占比，应给阶段添加 NVTX 范围后重新采集。

PyTorch Profiler 的 CUPTI 问题也单独记录：上游报告中有 RTX 5060 Laptop GPU 与 CUDA 12.8 上的同类 `CUPTI_ERROR_INVALID_DEVICE` 现象；本实验改用 Nsight Systems 成功取得 CUDA trace，没有为获得 trace 而更换已经固定的 PyTorch 环境。[对应的 PyTorch 上游问题](https://github.com/pytorch/pytorch/issues/174195)

## OperatorEngine Prefill Attention 替换实验

### 接入范围与接口检查

实验只替换 Prefill 阶段的 Attention，模型其余部分和 Decode 阶段保持原实现。GPT-2 Small 的 Attention 输入整理为 `[B, 12, prompt_len, 64]`，Q/K/V 序列长度相同，符合 OperatorEngine Attention 的当前接口；本次使用 `provider="auto"`，运行信息确认 causal Prefill 选择了 Triton provider。Attention 输入由转置得到，进入算子前需转成 contiguous；这三次 Q/K/V 连续化拷贝计入端到端计时。

Decode 的 Q shape 是 `[B, 12, 1, 64]`，而 K/V shape 是 `[B, 12, cached_len, 64]`。Q/K/V 序列长度不同，因此不能直接走当前 OperatorEngine 接口。本实验没有替换 Decode 算子。

### 数值与生成结果

在 FP16 下，数值检查只比较两条路径仍拥有相同生成历史时的 logits。若某一步采样出的 token 不同，后续上下文随即分叉，后续 logits 不再作为同一输入下的算子误差比较。设定的 `rtol=2e-2, atol=2e-2` 检查通过。

| Batch | Prompt | Output | 共享上下文最大/平均绝对误差 | 输出共同前缀 | token 全部一致 |
|---:|---:|---:|---:|---:|:---:|
| 1 | 32 | 8 | 0.0625 / 0.0242982 | 8 / 8 | 是 |
| 1 | 128 | 8 | 0.09375 / 0.0177115 | 2 / 8 | 否 |

Prompt=128 时，两条路径的前两个生成 token 相同，在第三个 token 分叉：PyTorch 选择 token `240`，OperatorEngine 路径选择 token `95`。该步 PyTorch logits 的 top-1 与 top-2 间隔为 `0.03125`，OperatorEngine 输出的前两名间隔为 `0`。这说明本例中候选值非常接近，FP16 数值差异足以改变贪心采样结果；通过容差检查并不等于自回归生成序列必然逐 token 一致。

### 端到端时延与显存

以下数据来自 5 次 warmup、30 轮成对测量。计时包括模型完整生成 8 个 token 的过程，以及 OperatorEngine 接入时 Q/K/V 的 contiguous 拷贝；每条路径均采用预分配 KV Cache。

| Batch | Prompt | Output | PyTorch p50 | OperatorEngine p50 | Speedup | PyTorch tokens/s | OperatorEngine tokens/s | Peak extra |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 8 | 27.729 ms | 27.691 ms | 1.001x | 288.51 | 288.90 | 4.73 MiB，两者相同 |
| 1 | 128 | 8 | 33.964 ms | 34.628 ms | 0.981x | 235.54 | 231.03 | 17.25 MiB，两者相同 |

当前两组 case 均没有显示稳定的端到端加速：prompt=32 基本持平，prompt=128 的替换路径慢约 1.9%，峰值额外显存相同。此前同一短 benchmark 的 p50 曾明显变化，因此这组单次 30 轮测量可作为本机结果记录，但不足以证明几个百分点内的差异具有统计显著性。整体路径还包含 Q/K/V 连续化和 Python/算子调度成本；Prefill Attention 的局部实现即使更快，也可能被这些端到端成本抵消。

### 复现实验命令

从 `01_gpt2_inference/` 项目根目录运行：

```bash
python3 benchmarks/benchmark_operatorengine.py \
  --batch-sizes 1 \
  --prompt-lengths 32,128 \
  --output-lengths 8 \
  --warmup-runs 5 \
  --measured-runs 30
```

## 结论

- 已完成多组 batch、prompt length、output length 的 Prefill、TTFT、TPOT、step p95、总 tokens/s 和显存扫描。
- Peak extra 随 batch 与序列长度增长的趋势清楚；短时延数据噪声较大，应作为本机基线而非普遍规律。
- Nsight Systems 已提供真实 CUDA kernel/runtime 证据；本次 kernel 汇总以 cuBLAS GEMV 变体为主。
- runtime API 时间、kernel 时间和 trace 总跨度是不同口径，不能直接相加或互换。
- PyTorch Profiler CUDA 活动初始化失败已记录；Nsight trace 成功，使本节 profiling 学习目标达成。
- OperatorEngine Prefill Attention 接口接入和数值/性能测量已完成；Decode 因 Q/K/V 长度不一致留在 PyTorch。当前测试 shape 下端到端基本持平，没有证据证明稳定加速。
- FP16 logits 通过容差但 prompt=128 的生成在第三个 token 分叉；top-1/top-2 间隔表明采样对该步数值扰动敏感，后续不同上下文的 logits 不应直接比较。

## 复现实验命令

从 `01_gpt2_inference/` 项目根目录运行 sweep：

```bash
python3 benchmarks/benchmark_sweep.py \
  --batch-sizes 1,2 \
  --prompt-lengths 32,128 \
  --output-lengths 8,32 \
  --dtype fp16 \
  --warmup-runs 3 \
  --measured-runs 10
```

Profiler 采集的单组 case 应使用 Nsight Systems 单独运行，并把 profiler 结果与基准计时分开保存。原始 `.nsys-rep` 与 `.sqlite` 文件保留在 `profiles/` 目录，以便后续审阅和复算。
