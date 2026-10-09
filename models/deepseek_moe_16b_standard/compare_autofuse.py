# coding=utf-8
"""Fixed-shape DeepSeek-MoE 16B comparison for eager, Inductor, and AutoFuse."""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoTokenizer

from .configuration_deepseek_moe_16b import DeepseekMoeConfig
from .modeling_deepseek_moe_16b import DeepseekMoeForCausalLM


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
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=None,
        help="Optional layer cap for compile smoke tests.",
    )
    args = parser.parse_args()

    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable")
    AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    config = DeepseekMoeConfig.from_pretrained(args.model_path)
    if args.num_hidden_layers is not None:
        config.num_hidden_layers = args.num_hidden_layers
    model = DeepseekMoeForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
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
    print(f"num_hidden_layers={config.num_hidden_layers}")
    print(f"avg_forward_ms={elapsed * 1000 / args.steps:.3f}")


if __name__ == "__main__":
    main()
