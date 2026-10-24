"""5.3.3 固定 Token 长度的混合请求准备：只读本地分词器，不发送请求。

使用同一固定 revision，生成 /v1/completions 的 Token ID 输入。
这是合成性能工作量，不用于评价回答质量；聊天模板不参与本组实验。
"""

import argparse
import json
from pathlib import Path


def make_exact_prompt(seed_ids: list[int], target_len: int) -> list[int]:
    """重复合法 seed_ids 并截断，返回恰好 target_len 个 ID 的新列表。

    输入列表非空、元素为非负整数；target_len 为正整数。
    拒绝非法输入时抛 ValueError，不修改原列表。
    不先解码再重新分词，否则不能保证最终 ID 序列长度和内容不变。
    """
    # 先校验，再向上取整计算重复次数，最后截取目标长度。
    if type(target_len) is not int or target_len <= 0:
        raise ValueError("target_len 必须为正整数")

    # 空列表无法提供内容，也会导致后面的除法除以零。
    if not seed_ids:
        raise ValueError("seed_ids 不能为空")

    if any(type(token_id) is not int or token_id < 0
           for token_id in seed_ids):
        raise ValueError("Token ID 必须为非负整数")
    
    # 向上取整，保证重复后的数量不少于目标长度。
    repeat_count = (target_len + len(seed_ids) - 1) // len(seed_ids)

    # 切片返回新列表，不修改原列表。
    # 得到 ID 后不要再解码并重新编码，否则不能保证序列保持不变。
    return (seed_ids * repeat_count)[:target_len]


def build_requests(tokenizer, model="qwen2.5-1.5b", max_model_len=2048) -> list[dict]:
    """组装请求和预期计数；长度构造仍调用你实现的 make_exact_prompt。"""
    requests = []
    for cohort, count, isl, osl in [("short", 4, 128, 256), ("long", 3, 1536, 32)]:
        if isl + osl > max_model_len:
            raise ValueError("候选工作量超过 max_model_len，不能默默截断")
        for index in range(count):
            request_id = f"{cohort}_{index}"
            # 不同请求在输入开头就有区别；普通基线还必须关闭 prefix caching。
            # 本组不会用“长共同前缀 + 不同后缀”模拟真实前缀复用场景。
            seed_text = f"请求 {request_id}。请解释计算、显存和请求调度如何影响大模型推理性能。"
            seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
            prompt_ids = make_exact_prompt(seed_ids, isl)
            if not isinstance(prompt_ids, list) or len(prompt_ids) != isl:
                raise ValueError("构造结果不是预期长度的 Token ID 列表")
            requests.append({
                "request_id": request_id,
                "cohort": cohort,
                "expected_prompt_tokens": isl,
                "expected_completion_tokens": osl,
                "body": {
                    "model": model,
                    # 直接传整数列表，不把它还原成文本后再让服务器重新分词。
                    "prompt": prompt_ids,
                    "add_special_tokens": False,
                    "n": 1,
                    "temperature": 0.0,
                    "max_tokens": osl,
                    # max_tokens 单独只是上限。合成长短输出需要抑制 EOS 提前停止；
                    # 真实执行后仍须核对 usage，而不是相信这里的预期值。
                    "ignore_eos": True,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            })
    return requests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-spec", type=Path, default=(
        Path(__file__).resolve().parents[2] / "02_single_gpu_benchmark" / "model_spec.json"))
    parser.add_argument("--model", default="qwen2.5-1.5b")
    parser.add_argument("--max-model-len", type=int, default=2048)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出文件已存在，请选择新文件名；不会覆盖旧工作量")

    # 只在实际准备文件时加载现有依赖；单元测试用小分词器替身，不加载模型。
    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers import AutoTokenizer

    spec = json.loads(args.model_spec.read_text(encoding="utf-8"))
    model_path = (Path(HF_HUB_CACHE) / ("models--" + spec["model_id"].replace("/", "--"))
                  / "snapshots" / spec["revision"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False)
    requests = build_requests(tokenizer, args.model, args.max_model_len)
    prepared = {
        "purpose": "synthetic_fixed_token_lengths_not_quality_evaluation",
        "api_path": "/v1/completions",
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "tokenizer_path": str(model_path),
        "tokenizer_class": type(tokenizer).__name__,
        "requirements": {"enable_prefix_caching": False,
                         "max_model_len": args.max_model_len},
        "release_policy": "send shorts; after every short first content, release longs",
        "requests": requests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(prepared, output, ensure_ascii=False, indent=2)
            output.write("\n")
    except FileExistsError:
        parser.error("输出文件已存在，请选择新文件名")
    print("WORKLOAD_PREPARED_NOT_SENT")
    print(f"tokenizer: {type(tokenizer).__name__}; revision: {spec['revision']}")
    for request in requests:
        print(f"{request['request_id']}: input={len(request['body']['prompt'])}, "
              f"expected_output={request['expected_completion_tokens']}")
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
