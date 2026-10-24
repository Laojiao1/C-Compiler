#!/usr/bin/env bash
# 顺序管理本脚本自己创建的服务；不接管或结束用户已有的引擎。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run_dir="${1:-stage_3_serving/results/cache_validation_20261023}"
modes=(capacity prefix_off prefix_on)
if (( $# > 1 )); then
    modes=("${@:2}")
fi
for mode in "${modes[@]}"; do
    case "$mode" in capacity|prefix_off|prefix_on) ;; *) echo "未知模式：$mode" >&2; exit 2 ;; esac
done
python_path=/home/laojiao/miniconda3/envs/vllm/bin/python
if [[ -e "$run_dir" ]]; then
    echo "运行目录已存在，停止，避免覆盖证据：$run_dir" >&2
    exit 2
fi
if ss -H -ltn 'sport = :8000' | grep -q . || pgrep -af '[v]llm serve|[E]ngineCore'; then
    echo "已有服务或端口监听；不自动接管，停止执行。" >&2
    exit 2
fi
mkdir -p "$run_dir"
owned_pid=""
stop_owned() {
    if [[ -n "$owned_pid" ]]; then
        if kill -0 "$owned_pid" 2>/dev/null; then
            kill -INT "$owned_pid"
            for ((i=0; i<60; i++)); do
                if ! kill -0 "$owned_pid" 2>/dev/null; then break; fi
                sleep 1
            done
            if kill -0 "$owned_pid" 2>/dev/null; then
                kill -TERM "$owned_pid"
            fi
        fi
        wait "$owned_pid" 2>/dev/null || true
        owned_pid=""
    fi
}
trap stop_owned EXIT
trap 'exit 130' INT TERM
"$python_path" -m stage_3_serving.cache_experiment prepare --output "$run_dir/workload.json"
for mode in "${modes[@]}"; do
    echo "START $mode"
    bash stage_3_serving/serve_stream_probe.sh "$mode" > "$run_dir/${mode}_server.log" 2>&1 &
    owned_pid=$!
    deadline=$((SECONDS + 480))
    while true; do
        if ! kill -0 "$owned_pid" 2>/dev/null; then
            echo "服务启动退出，查看 $run_dir/${mode}_server.log" >&2
            exit 1
        fi
        if curl --noproxy '*' -fsS --connect-timeout 1 --max-time 2 -o /dev/null http://127.0.0.1:8000/health 2>/dev/null &&
           curl --noproxy '*' -fsS --connect-timeout 1 --max-time 2 -o /dev/null http://127.0.0.1:8000/metrics 2>/dev/null; then
            break
        fi
        if (( SECONDS >= deadline )); then
            echo "服务未在 480 秒内就绪，保留日志后停止。" >&2
            exit 1
        fi
        sleep 1
    done
    "$python_path" -m stage_3_serving.cache_experiment run --mode "$mode" \
        --workload "$run_dir/workload.json" --output "$run_dir/${mode}.json"
    stop_owned
    for ((i=0; i<60; i++)); do
        if ! pgrep -f '[E]ngineCore' >/dev/null && ! ss -H -ltn 'sport = :8000' | grep -q .; then break; fi
        sleep 1
    done
    if pgrep -af '[v]llm serve|[E]ngineCore' || ss -H -ltn 'sport = :8000' | grep -q .; then
        echo "旧服务尚未完全退出，不启动下一引擎。" >&2
        exit 1
    fi
    echo "FINISHED $mode"
done
echo "CACHE_EXPERIMENTS_PASS: $run_dir"
