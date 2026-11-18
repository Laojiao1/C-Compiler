# 大模型量化实验

从 WSL2 的 `quant` 根目录运行下面的命令。结果和验收见 [实验记录](./实验记录.md)。

```bash
cd /mnt/e/Code/C++Compiler/5_LLM_Serving/quant
conda activate triton
```

## 单层权重量化（步骤 1–3）

采集脚本和现有的 `down_proj_sample.pt` 都在 `weight_quant/`。下面的对照直接使用现有采集数据。

```bash
python -m weight_quant.int4_per_tensor
python -m weight_quant.int4_per_group
python -m weight_quant.smoothquant_check
python -m weight_quant.gptq_toy
python -m weight_quant.gptq_slice
python -m weight_quant.awq_linear
```

需要重新采集时，运行 `python -m weight_quant.capture_down_proj`。默认写到采集脚本旁边，可用 `--output` 指定其他保存位置。

## W4A16 算子（步骤 4–5）

`w4a16/marlin_w4a16.cu` 是主实验。可以用下面的命令重新编译和验证。已有的可执行文件在 `w4a16/bin/`，早期 Triton 练习 `w4a16_gemm.py` 也在 `w4a16/`。

```bash
/usr/local/cuda/bin/nvcc -arch=sm_120 -O3 -std=c++17 \
    w4a16/marlin_w4a16.cu -o /tmp/marlin_w4a16
/tmp/marlin_w4a16
```

复跑早期 Triton 练习用 `python w4a16/w4a16_gemm.py --kernel`。

## FP8 与 FP4 格式（步骤 6）

```bash
python -m formats.fp8_scaling
python -m formats.fp4_block_scaling
```

## FP8 KV Cache（步骤 7）

依次检查缓存写入、K/V 分项误差，以及 Triton decode 和容量。

```bash
python -m kv_cache.fp8_kv_cache
python -m kv_cache.fp8_kv_attention
python -m kv_cache.fp8_kv_decode_triton
```

`archive/` 存放早期的 vLLM 启动验证脚本。
