"""5.2.1 单请求功能验证：核心推理代码按学习步骤逐段补充。"""

import json
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE
from vllm import LLM


def main() -> None:
    spec = json.loads(Path(__file__).with_name("model_spec.json").read_text(encoding="utf-8"))
    cache_name = "models--" + spec["model_id"].replace("/", "--")
    model_path = Path(HF_HUB_CACHE) / cache_name / "snapshots" / spec["revision"]
    if not (model_path / spec["weight_file"]).is_file():
        raise FileNotFoundError("模型尚未准备好，请先完成 prepare_model.py")
    print(f"Model path: {model_path}", flush=True)

    # 第一步：创建 LLM，引擎初始化成功后打印 ENGINE_INIT_PASS。
    llm = LLM(
        model = str(model_path), # 把路径对象转为接口需要的字符串，从本地加载
        dtype = "bfloat16",
        tensor_parallel_size=1,
        max_model_len = 2048,
        gpu_memory_utilization = 0.70, # 引擎显存预算起点，调整为 0.75
        enable_prefix_caching = False, # 不采用 prefix cache
        generation_config = "vllm",
        max_num_seqs = 16, # 单步最多调度的序列数
        max_num_batched_tokens = 2048, # 单步 Token 计算预算
    )
    print("ENGINE_INIT_PASS", flush=True)
    # 后续再添加单请求输入、采样参数、generate 和输出长度核对。


if __name__ == "__main__":
    # vLLM 可能创建 worker 子进程，避免子进程导入本文件时重复运行 main。
    main()
