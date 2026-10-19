# GPT-2 单卡推理实验环境

## 当前已确认环境

记录日期：2026-09-22

| 项目 | 值 |
|---|---|
| OS / 运行环境 | WSL Ubuntu 20.04 |
| Python | 3.10.20 |
| Conda 环境 | `triton` |
| PyTorch | `2.11.0+cu128` |
| PyTorch CUDA runtime | 12.8 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| GPU 显存 | 8151 MiB |
| NVIDIA 驱动 | 576.65 |
| Driver reported CUDA | 12.9 |
| Transformers | `5.17.0` |

## 版本边界

本阶段固定使用 WSL 中的 `triton` 环境。Windows Conda 环境中的 PyTorch 不作为本实验环境：它无法为 RTX 5060（`sm_120`）执行 CUDA kernel。

## CUDA smoke test

在安装 `transformers` 前，先运行以下命令确认 CUDA kernel、同步和显存统计可用：

```bash
python3 - <<'PY'
import torch

assert torch.cuda.is_available()
device = torch.device("cuda")
x = torch.randn((1024, 1024), device=device, dtype=torch.float16)
torch.cuda.synchronize()
y = x @ x
torch.cuda.synchronize()
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("checksum:", float(y.float().mean().cpu()))
print("max allocated:", torch.cuda.max_memory_allocated())
PY
```

## GPT-2 依赖

当前环境已安装并确认：

```bash
python3 -c "import transformers; print(transformers.__version__)"
```

输出版本：`5.17.0`。

模型固定为 Hugging Face `gpt2`。数值对齐阶段先使用 `float32`，性能阶段再单独记录 `float16`，不能混用两组结果。

## 统一计时边界

- 不计入 tokenizer、CPU 到 GPU 的输入拷贝和 sampling。
- 输入张量已位于 GPU 后再开始计时。
- 计时前后都调用 `torch.cuda.synchronize()`。
- 每组实验先 warmup 10 次，再测量 50 次。
- 保存 median、p50、p95、tokens/s 和峰值 `torch.cuda.max_memory_allocated()`。

## Benchmark 入口

两个性能脚本集中放在 `benchmarks/` 子目录，模型、Decode 和测试仍留在项目根目录。请从 `01_gpt2_inference/` 根目录运行：

```bash
python3 benchmarks/benchmark_decode.py
python3 benchmarks/benchmark_sweep.py \
  --batch-sizes 1,2 \
  --prompt-lengths 32,128 \
  --output-lengths 8,32 \
  --dtype fp16 \
  --warmup-runs 3 \
  --measured-runs 10
```

若要为第一组 shape 另采 profiler trace，在扫描命令末尾加上 `--profile --trace-dir profiles`。Profiler 采样会增加开销，trace 用于观察算子，不作为延迟数据。
