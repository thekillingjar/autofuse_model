# coding=utf-8
"""Fixed-shape BERT/ViT comparison for eager, Inductor, and AutoFuse."""

from __future__ import annotations

import argparse
import time

import torch

from .modeling_transformers import MODEL_BUILDERS, build_model


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
    parser.add_argument("--model-type", choices=sorted(MODEL_BUILDERS), required=True)
    parser.add_argument("--mode", choices=("none", "inductor", "autofuse"), default="none")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-hidden-layers", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable")
    shape_arg = args.seq_len if args.model_type == "bert" else args.image_size
    model = build_model(args.model_type, args.num_hidden_layers, shape_arg).eval().to(torch.bfloat16).to("npu")
    model = compile_model(model, args.mode)

    if args.model_type == "bert":
        inputs = torch.randint(
            0,
            30522,
            (args.batch_size, args.seq_len),
            dtype=torch.long,
            device="npu",
        )
        attention_mask = torch.ones_like(inputs)

        def run_once():
            return model(inputs, attention_mask=attention_mask)

    else:
        inputs = torch.randn(
            args.batch_size,
            3,
            args.image_size,
            args.image_size,
            dtype=torch.bfloat16,
            device="npu",
        )

        def run_once():
            return model(inputs)

    with torch.inference_mode():
        for _ in range(args.warmup):
            run_once()
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(args.steps):
            run_once()
        torch.npu.synchronize()

    elapsed = time.perf_counter() - start
    print(f"model_type={args.model_type}")
    print(f"mode={args.mode}")
    print(f"batch_size={args.batch_size}")
    if args.model_type == "bert":
        print(f"seq_len={args.seq_len}")
    else:
        print(f"image_size={args.image_size}")
    print(f"num_hidden_layers={args.num_hidden_layers or 12}")
    print(f"avg_forward_ms={elapsed * 1000 / args.steps:.3f}")


if __name__ == "__main__":
    main()
