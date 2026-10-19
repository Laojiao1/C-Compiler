"""Compare GPT-2 memory estimates with PyTorch CUDA allocator readings."""

import argparse
from typing import List

import torch
from transformers import GPT2LMHeadModel

from model import GPT2ReferenceModel


MIB = 1024**2


def to_mib(num_bytes: int) -> float:
    return num_bytes / MIB


def registered_storage_bytes(model: torch.nn.Module) -> int:
    """按底层 storage 去重，统计模型已注册参数和 buffer 的真实存储量。"""
    seen_storages = set()
    total_bytes = 0

    tensors = list(model.named_parameters(remove_duplicate=False))
    tensors.extend(model.named_buffers(remove_duplicate=False))

    for _, tensor in tensors:
        storage = tensor.untyped_storage()
        storage_key = (tensor.device, storage.data_ptr())
        if storage_key not in seen_storages:
            seen_storages.add(storage_key)
            total_bytes += storage.nbytes()

    return total_bytes


def parse_int_list(value: str) -> List[int]:
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def load_model(device: torch.device, dtype: torch.dtype) -> GPT2ReferenceModel:
    """Load official GPT-2 weights into the hand-written model."""
    reference = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        torch_dtype=dtype,
    ).eval()

    model = GPT2ReferenceModel(reference.config)
    missing_keys, unexpected_keys = model.load_state_dict(
        reference.state_dict(),
        strict=False,
    )

    if missing_keys:
        raise RuntimeError(f"missing model weights: {missing_keys}")
    if any(not key.endswith("masked_bias") for key in unexpected_keys):
        raise RuntimeError(f"unexpected model weights: {unexpected_keys}")

    # 删除 CPU 参考模型，避免它占用主机内存；正式测量只看手写模型。
    del reference
    return model.to(device=device, dtype=dtype).eval()


def measure_dtype(
    dtype_name: str,
    dtype: torch.dtype,
    batch_size: int,
    sequence_lengths: List[int],
) -> None:
    device = torch.device("cuda")
    model = load_model(device, dtype)

    # parameters() 默认会去重，因此共享的 token embedding / LM head 只算一次。
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    buffer_bytes = sum(
        buffer.numel() * buffer.element_size()
        for buffer in model.buffers()
    )
    registered_bytes = registered_storage_bytes(model)

    torch.cuda.synchronize(device)
    model_allocated = torch.cuda.memory_allocated(device)

    print(f"\n=== dtype: {dtype_name} ===")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"parameter bytes (theory): {to_mib(parameter_bytes):.2f} MiB")
    print(f"registered buffer bytes: {to_mib(buffer_bytes):.2f} MiB")
    print(f"unique registered tensor storage: {to_mib(registered_bytes):.2f} MiB")
    print(f"model allocated by PyTorch: {to_mib(model_allocated):.2f} MiB")
    print(
        "allocated minus registered tensor storage: "
        f"{to_mib(model_allocated - registered_bytes):.2f} MiB"
    )
    print(
        "embedding/LM-head share storage: "
        f"{model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()}"
    )

    for seq_len in sequence_lengths:
        if seq_len > model.config.n_positions:
            raise ValueError(
                f"sequence length {seq_len} exceeds GPT-2 limit "
                f"{model.config.n_positions}"
            )

        input_ids = torch.randint(
            low=0,
            high=model.config.vocab_size,
            size=(batch_size, seq_len),
            dtype=torch.long,
            device=device,
        )

        # 先跑一次让 CUDA 和底层矩阵运算完成初始化；这次结果不纳入峰值测量。
        with torch.inference_mode():
            warmup_logits = model(input_ids)
        torch.cuda.synchronize(device)
        del warmup_logits

        # baseline 包含模型权重、固定 buffer 和 input_ids。
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)

        # inference_mode 关闭梯度记录，符合推理时的内存使用方式。
        with torch.inference_mode():
            logits = model(input_ids)
        torch.cuda.synchronize(device)

        forward_peak_allocated = torch.cuda.max_memory_allocated(device)
        forward_peak_reserved = torch.cuda.max_memory_reserved(device)
        forward_live_allocated = torch.cuda.memory_allocated(device)

        kv_bytes = (
            2
            * model.config.n_layer
            * batch_size
            * seq_len
            * model.config.n_embd
            * torch.empty((), dtype=dtype).element_size()
        )

        # 用一个与实际 Cache 总元素数相同的张量，验证分配器报告的增量。
        # 两个最前面的维度分别表示 K/V 和 Transformer 层。
        cache_before = torch.cuda.memory_allocated(device)
        kv_cache = torch.empty(
            2,
            model.config.n_layer,
            batch_size,
            model.config.n_head,
            seq_len,
            model.config.n_embd // model.config.n_head,
            dtype=dtype,
            device=device,
        )
        torch.cuda.synchronize(device)
        cache_after = torch.cuda.memory_allocated(device)
        measured_kv_bytes = cache_after - cache_before

        print(
            f"T={seq_len:4d}, B={batch_size} | "
            f"KV theory={to_mib(kv_bytes):8.2f} MiB, "
            f"KV allocated delta={to_mib(measured_kv_bytes):8.2f} MiB | "
            f"forward extra peak="
            f"{to_mib(forward_peak_allocated - baseline_allocated):8.2f} MiB | "
            f"forward live output="
            f"{to_mib(forward_live_allocated - baseline_allocated):8.2f} MiB | "
            f"peak reserved={to_mib(forward_peak_reserved):8.2f} MiB "
            f"(baseline {to_mib(baseline_reserved):.2f} MiB)"
        )

        del kv_cache, logits, input_ids
        torch.cuda.synchronize(device)

    del model
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure GPT-2 weights, KV Cache, and forward peak memory."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--sequence-lengths",
        type=parse_int_list,
        default=parse_int_list("128,512,1024"),
    )
    parser.add_argument(
        "--dtype",
        choices=("fp32", "fp16", "both"),
        default="both",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this script in the WSL GPU env.")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"batch size: {args.batch_size}")
    print(f"sequence lengths: {args.sequence_lengths}")
    print("MiB uses 1 MiB = 1024 * 1024 bytes.")

    dtypes = {
        "fp32": torch.float32,
        "fp16": torch.float16,
    }
    dtype_names = ("fp32", "fp16") if args.dtype == "both" else (args.dtype,)

    for dtype_name in dtype_names:
        measure_dtype(
            dtype_name,
            dtypes[dtype_name],
            args.batch_size,
            args.sequence_lengths,
        )


if __name__ == "__main__":
    main()
