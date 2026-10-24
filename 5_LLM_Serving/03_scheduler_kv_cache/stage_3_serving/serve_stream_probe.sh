#!/usr/bin/env bash
# 不带参数时保留已经验证过的预跑配置；A/B/C 用于正式单变量对照。
# 从 5.3 代码根目录运行：bash stage_3_serving/serve_stream_probe.sh A
# 必须在 WSL 执行；本脚本不安装依赖，也不修改 shell 的全局环境。
set -euo pipefail

# 三个配置只改变预算或分块开关，KV 池与物理块大小保持一致。
# 1 GiB 不是最优值：它给本轮按块估计约 171 MiB 的 KV 载荷留出余量，
# 同时比原自动分配的约 2.25 GiB 更保守；是否启动成功仍以本机为准。
case "${1:-probe}" in
    probe) budget=2048; chunk_flag=--enable-chunked-prefill; extra_cache_args=() ;;
    A) budget=512; chunk_flag=--enable-chunked-prefill
       extra_cache_args=(--kv-cache-memory-bytes 1073741824 --block-size 16) ;;
    B) budget=4096; chunk_flag=--enable-chunked-prefill
       extra_cache_args=(--kv-cache-memory-bytes 1073741824 --block-size 16) ;;
    C) budget=4096; chunk_flag=--no-enable-chunked-prefill
       extra_cache_args=(--kv-cache-memory-bytes 1073741824 --block-size 16) ;;
    capacity) budget=512; chunk_flag=--enable-chunked-prefill
       extra_cache_args=(--kv-cache-memory-bytes 134217728 --block-size 16) ;;
    prefix_off|prefix_on) budget=512; chunk_flag=--enable-chunked-prefill
       extra_cache_args=(--kv-cache-memory-bytes 1073741824 --block-size 16) ;;
    -h|--help) echo "用法：bash stage_3_serving/serve_stream_probe.sh [probe|A|B|C|capacity|prefix_off|prefix_on]"; exit 0 ;;
    *) echo "未知实验配置。" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    echo "只接受一个配置名称。" >&2
    exit 2
fi
prefix_flag=--no-enable-prefix-caching
if [[ "${1:-probe}" == prefix_on ]]; then
    prefix_flag=--enable-prefix-caching
fi

model_path="/home/laojiao/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
vllm_command="/home/laojiao/miniconda3/envs/vllm/bin/vllm"
ptxas_path="/home/laojiao/miniconda3/envs/vllm/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas"

if [[ ! -f "$model_path/model.safetensors" || ! -x "$vllm_command" || ! -x "$ptxas_path" || ! -d /usr/local/cuda-12.8/bin ]]; then
    echo "已验证路径缺失，停止启动；请核对路径，不要据此重装环境。" >&2
    exit 1
fi

# 不替用户终止任何进程。发现已有引擎或端口占用时先停下来核对。
if ss -H -ltn 'sport = :8000' | grep -q .; then
    echo "8000 端口已有监听，停止重复启动；先核对已有服务。" >&2
    exit 1
fi
if pgrep -af '[v]llm serve|[v]llm.entrypoints|[E]ngineCore'; then
    echo "发现疑似已有引擎，停止启动，避免争抢显存。" >&2
    exit 1
fi

# 只在当前脚本及其替换出的服务进程中生效，保留历史已验证工具链。
export PATH="/usr/local/cuda-12.8/bin:$PATH"
export TRITON_PTXAS_BLACKWELL_PATH="$ptxas_path"

exec "$vllm_command" serve "$model_path" \
    --served-model-name qwen2.5-1.5b \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.70 \
    --max-num-seqs 16 \
    --max-num-batched-tokens "$budget" \
    "$chunk_flag" \
    "${extra_cache_args[@]}" \
    "$prefix_flag" \
    --generation-config vllm
