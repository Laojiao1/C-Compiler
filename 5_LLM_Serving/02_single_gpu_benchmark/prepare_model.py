"""下载固定版本并核对权重；不加载推理引擎，也不测吞吐。"""

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def main() -> None:
    spec_path = Path(__file__).with_name("model_spec.json")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    print(f"Model: {spec['model_id']}", flush=True)
    print(f"Revision: {spec['revision']}", flush=True)

    # 用完整 commit 下载；保留官方缓存，重复执行可以复用已完成内容。
    snapshot = Path(snapshot_download(
        repo_id=spec["model_id"],
        revision=spec["revision"],
        allow_patterns=spec["files"],
        max_workers=2,
        token=False,
    ))
    missing = [name for name in spec["files"] if not (snapshot / name).is_file()]
    if missing:
        raise RuntimeError(f"模型文件缺失：{missing}")

    weight = snapshot / spec["weight_file"]
    if weight.stat().st_size != spec["weight_bytes"]:
        raise RuntimeError("权重文件大小与官方记录不一致")

    print("Checking weight SHA-256...", flush=True)
    digest = hashlib.sha256()
    # 分块校验，避免一次把约 3 GB 权重读进 CPU 内存。
    with weight.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != spec["weight_sha256"]:
        raise RuntimeError("权重 SHA-256 与官方记录不一致")

    # 验证 tokenizer 文件已齐全；只读本地文件，不额外下载。
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    messages = [{"role": "user", "content": "请用一句话解释什么是 KV Cache。"}]
    token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    print(f"Snapshot: {snapshot}", flush=True)
    print(f"Weight bytes: {weight.stat().st_size}", flush=True)
    print(f"Weight SHA-256: {digest.hexdigest()}", flush=True)
    print(f"Tokenizer size: {len(tokenizer)}", flush=True)
    print(f"Smoke prompt tokens: {len(token_ids)}", flush=True)
    print("MODEL_PREPARATION_PASS", flush=True)


if __name__ == "__main__":
    main()
