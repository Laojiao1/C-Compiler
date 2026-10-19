"""5.2.1 单请求功能验证：核心推理代码按学习步骤逐段补充。"""
"""
编译指令：
env \
  PATH="/usr/local/cuda-12.8/bin:$PATH" \
  TRITON_PTXAS_BLACKWELL_PATH="/home/laojiao/miniconda3/envs/vllm/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas" \
  python -u offline_smoke.py
"""

import json
import time
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE
from vllm import LLM, SamplingParams


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
        gpu_memory_utilization = 0.70, # 引擎显存预算起点，调整为 0.70
        enable_prefix_caching = False, # 不采用 prefix cache
        generation_config = "vllm",
        max_num_seqs = 16, # 单步最多调度的序列数
        max_num_batched_tokens = 2048, # 单步 Token 计算预算
    )
    print("ENGINE_INIT_PASS", flush=True)

    # # 添加单请求输入、采样参数、generate 和输出长度核对。
    tokenizer = llm.get_tokenizer() # 获取引擎使用的分词器
    # messages = [
    #     {
    #         "role": "user",
    #         "content": "请用一句话解释什么是 KV Cache。"
    #     }
    # ]
    # prompt_ids = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=True,
    #     add_generation_prompt=True,
    #     return_dict=False, # 要求返回 ID 列表，避免把字典字段数误认为 Token 数
    # )
    # prompts = [{"prompt_token_ids": prompt_ids}]

    target_isl = 512
    target_osl = 128

    seed_ids = tokenizer.encode(
        "大模型推理需要关注计算、显存和请求调度。",
        add_special_tokens=False,
    )

    # 先得到一小段合法 Token ID，重复到足够长，再截取前 512 个
    repeat_count = (target_isl + len(seed_ids) - 1) // len(seed_ids)
    prompt_ids = (seed_ids * repeat_count)[:target_isl]
    # prompts = [{"prompt_token_ids": prompt_ids}]
    num_requests = 64
    prompts = [
        {"prompt_token_ids": prompt_ids.copy()}
        for _ in range(num_requests)
    ]

    assert len(prompt_ids) == target_isl

    # 生成一个 greedy 回答，忽略 EOS，最多生成 128 Token，不转换输出文字
    sampling_params = SamplingParams(
        n=1, # 候选回答数
        temperature=0.0, # 是否采用 greedy 选择
        # max_tokens=32, # 最多生成的 token 数
        max_tokens=target_osl, 
        ignore_eos=True, # 是否忽略 EOS
        detokenize=False, # 是否转换输出文字
    )

    warmup_runs = 2
    measure_runs = 5

    for i in range(warmup_runs):
        warmup_results = llm.generate(prompts, sampling_params, use_tqdm=False)
        assert len(warmup_results) == num_requests
        
        # assert len(warmup_results) == 1
        # assert warmup_results[0].finished
        # assert len(warmup_results[0].prompt_token_ids) == target_isl
        # assert len(warmup_results[0].outputs[0].token_ids) == target_osl

        for result in warmup_results:
            assert result.finished
            assert len(result.prompt_token_ids) == target_isl
            assert len(result.outputs) == 1
            assert len(result.outputs[0].token_ids) == target_osl


        print(f"Warmup {i + 1}/{warmup_runs} done")

    elapsed_samples = []

    for i in range(measure_runs):
        print(f"\nMeasure {i + 1}/{measure_runs}")

        # 将现有 start = time.perf_counter() 开始，
        # 到四种吞吐打印结束的完整代码块放在这里。
        start = time.perf_counter()
        results = llm.generate(prompts, sampling_params, use_tqdm=False)
        elapsed_s = time.perf_counter() - start
    
        result = results[0]
        output = result.outputs[0]
    
        print("Request count:", len(results))
        print("Finished:", result.finished)
        print("Prompt tokens:", len(result.prompt_token_ids))
        print("Output tokens:", len(output.token_ids))
        print("Finish reason:", output.finish_reason)
    
        # print("Text:", output.text)
    
        # assert len(results) == 1
        # assert result.finished
        # assert len(result.prompt_token_ids) == target_isl
        # assert len(output.token_ids) == target_osl

        assert len(results) == num_requests
        for result in results:
            assert result.finished
            assert len(result.prompt_token_ids) == target_isl
            assert len(result.outputs) == 1
            assert len(result.outputs[0].token_ids) == target_osl

        print("EXACT_LENGTH_PASS")
    
        assert elapsed_s > 0
    
        request_count = len(results)
        input_tokens = sum(len(r.prompt_token_ids) for r in results)
        output_tokens = sum(len(r.outputs[0].token_ids) for r in results)

        print("Total input tokens:", input_tokens)
        print("Total output tokens:", output_tokens) 
    
        print(f"Elapsed: {elapsed_s:.6f} s")
        print(f"Request throughput: {request_count / elapsed_s:.3f} req/s")
        print(f"Input throughput: {input_tokens / elapsed_s:.3f} tokens/s")
        print(f"Output throughput: {output_tokens / elapsed_s:.3f} tokens/s")
        print(f"Total throughput: {(input_tokens + output_tokens) / elapsed_s:.3f} tokens/s")

        elapsed_samples.append(elapsed_s)

    print("Elapsed samples:", elapsed_samples)


if __name__ == "__main__":
    # vLLM 可能创建 worker 子进程，避免子进程导入本文件时重复运行 main。
    main()
