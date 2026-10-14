# coding=utf-8
"""Fixed-shape classic model comparison for eager, Inductor, and AutoFuse."""

from __future__ import annotations

import argparse
import time

import torch

from .modeling_classic import MODEL_BUILDERS, build_model


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


def make_inputs(model_type: str, batch_size: int, device: str):
    if model_type == "afn":
        field_size = 39
        seq_len = field_size
        feat_ids = torch.randint(0, 32, (batch_size, seq_len)).to(device)
        feat_vals = torch.rand((batch_size, seq_len)).to(device)
        return feat_vals, feat_ids
    if model_type == "esmm":
        return (torch.randint(0, 10000, (batch_size, 20), dtype=torch.long, device=device),)
    if model_type == "dien":
        history = torch.randint(0, 10000, (batch_size, 64), dtype=torch.long, device=device)
        target = torch.randint(0, 10000, (batch_size,), dtype=torch.long, device=device)
        mask = torch.ones(batch_size, 64, dtype=torch.float32, device=device)
        return history, target, mask
    if model_type == "decision_transformer":
        seq_len = 32
        states = torch.randn(batch_size, seq_len, 17, device=device, dtype=torch.bfloat16)
        actions = torch.randn(batch_size, seq_len, 6, device=device, dtype=torch.bfloat16)
        returns_to_go = torch.randn(batch_size, seq_len, 1, device=device, dtype=torch.bfloat16)
        timesteps = torch.arange(seq_len, device=device).view(1, -1).expand(batch_size, -1)
        return states, actions, returns_to_go, timesteps
    if model_type == "sam":
        images = torch.randn(batch_size, 3, 256, 256, device=device, dtype=torch.bfloat16)
        points = torch.rand(batch_size, 4, 2, device=device, dtype=torch.bfloat16)
        return images, points
    raise ValueError(f"Unsupported model_type: {model_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=sorted(MODEL_BUILDERS), required=True)
    parser.add_argument("--mode", choices=("none", "inductor", "autofuse"), default="none")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable")
    model = build_model(args.model_type).eval().to(torch.bfloat16).to("npu")
    model = compile_model(model, args.mode)
    inputs = make_inputs(args.model_type, args.batch_size, "npu")

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(*inputs)
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(args.steps):
            model(*inputs)
        torch.npu.synchronize()

    elapsed = time.perf_counter() - start
    print(f"model_type={args.model_type}")
    print(f"mode={args.mode}")
    print(f"batch_size={args.batch_size}")
    print(f"avg_forward_ms={elapsed * 1000 / args.steps:.3f}")


if __name__ == "__main__":
    main()
