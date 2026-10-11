# coding=utf-8
"""Fixed-shape Gemma4-31B comparison for eager, Inductor, and AutoFuse."""

from __future__ import annotations

import argparse
import time

import torch

from .configuration_gemma4_31b import Gemma4TextConfig
from .modeling_gemma4_31b import Gemma4ForCausalLM


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
    parser.add_argument(
        "--disable-moe",
        action="store_true",
        help="Run dense MLP blocks only for isolating attention/MLP AutoFuse behavior.",
    )
    args = parser.parse_args()

    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable")
    config = Gemma4TextConfig.from_pretrained(args.model_path)
    if args.num_hidden_layers is not None:
        config.num_hidden_layers = args.num_hidden_layers
        config.layer_types = config.layer_types[: args.num_hidden_layers]
        if config.layer_types and config.layer_types[-1] != "full_attention":
            config.layer_types[-1] = "full_attention"
    if args.disable_moe:
        config.enable_moe_block = False
    model = Gemma4ForCausalLM(config).eval().to(torch.bfloat16).to("npu")
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
    print(f"enable_moe_block={config.enable_moe_block}")
    print(f"avg_forward_ms={elapsed * 1000 / args.steps:.3f}")


if __name__ == "__main__":
    main()
