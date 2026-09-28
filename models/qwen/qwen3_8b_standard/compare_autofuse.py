# coding=utf-8
"""Fixed-shape Qwen3-8B comparison for eager, Inductor, and AutoFuse."""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoTokenizer

from .configuration_qwen3_8b import Qwen3Config
from .modeling_qwen3_8b import Qwen3ForCausalLM


def compile_model(model, mode):
    if mode == "none":
        return model
    options = {"npu_backend": "ascendc"} if mode == "autofuse" else {}
    return torch.compile(
        model,
        backend="inductor",
        dynamic=False,
        fullgraph=True,
        options=options,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", choices=("none", "inductor", "autofuse"), default="none")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable")
    AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    config = Qwen3Config.from_pretrained(args.model_path)
    model = Qwen3ForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
    ).eval().to("npu")
    model = compile_model(model, args.mode)
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (args.batch_size, args.seq_len),
        dtype=torch.long,
        device="npu",
    )
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(args.steps):
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        torch.npu.synchronize()

    elapsed = time.perf_counter() - start
    print(f"mode={args.mode}")
    print(f"batch_size={args.batch_size}, seq_len={args.seq_len}")
    print(f"avg_forward_ms={elapsed * 1000 / args.steps:.3f}")


if __name__ == "__main__":
    main()
