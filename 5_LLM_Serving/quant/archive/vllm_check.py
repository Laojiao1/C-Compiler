import argparse

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="模型 ID 或本地快照目录")
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        dtype="float16",
        max_model_len=512,           # 这次只生成一小段文本。
        max_num_seqs=1,
        gpu_memory_utilization=0.60,
        enforce_eager=True,          # 首次启动省去 CUDA Graph 捕获。
    )

    params = SamplingParams(temperature=0.0, max_tokens=32)
    prompt = "请用一句话解释大模型推理中的 KV Cache。"
    request = llm.generate([prompt], params, use_tqdm=False)[0]
    output = request.outputs[0]
    print(f"Generated tokens: {len(output.token_ids)}")
    print(output.text)


if __name__ == "__main__":
    main()