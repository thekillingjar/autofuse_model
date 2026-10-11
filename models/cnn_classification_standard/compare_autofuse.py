# coding=utf-8
"""Fixed-shape CNN classification comparison for eager, Inductor, and AutoFuse."""

from __future__ import annotations

import argparse
import time

import torch

from .modeling_cnn import ARCH_BUILDERS, build_model


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
    parser.add_argument("--arch", choices=sorted(ARCH_BUILDERS), required=True)
    parser.add_argument("--mode", choices=("none", "inductor", "autofuse"), default="none")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable")
    model = build_model(args.arch, num_classes=args.num_classes).eval().to(torch.bfloat16).to("npu")
    model = compile_model(model, args.mode)
    images = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        device="npu",
        dtype=torch.bfloat16,
    )

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(images)
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(args.steps):
            model(images)
        torch.npu.synchronize()

    elapsed = time.perf_counter() - start
    print(f"arch={args.arch}")
    print(f"mode={args.mode}")
    print(f"batch_size={args.batch_size}, image_size={args.image_size}")
    print(f"avg_forward_ms={elapsed * 1000 / args.steps:.3f}")


if __name__ == "__main__":
    main()
