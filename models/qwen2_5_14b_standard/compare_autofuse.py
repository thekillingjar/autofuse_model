"""Fixed-shape, no-cache Qwen2.5-14B eager and AutoFuse benchmark."""

from __future__ import annotations

import argparse
import time

import torch

from .configuration_qwen2_5_14b import Qwen25_14BConfig
from .modeling_qwen2_5_14b import Qwen2ForCausalLM


def compile_model(model, mode):
    if mode in ("none", "eager"):
        return model
    if mode not in ("autofuse", "inductor"):
        raise ValueError(f"Unknown compile mode: {mode}")
    options = {"npu_backend": "ascendc"} if mode == "autofuse" else {}
    return torch.compile(
        model, backend="inductor", dynamic=False, fullgraph=True, options=options
    )


def benchmark(model, inputs, warmup, steps):
    with torch.inference_mode():
        torch.npu.synchronize()
        start = time.perf_counter()
        model(**inputs)
        torch.npu.synchronize()
        first_forward_ms = (time.perf_counter() - start) * 1000
        for _ in range(warmup):
            model(**inputs)
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            model(**inputs)
        torch.npu.synchronize()
        avg_forward_ms = (time.perf_counter() - start) * 1000 / steps
    return first_forward_ms, avg_forward_ms


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", help="Local HF checkpoint directory or HF model ID."
    )
    parser.add_argument(
        "--mode",
        choices=("none", "eager", "inductor", "autofuse", "compare"),
        default="none",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        help="Layer cap for smoke tests, not a full 14B benchmark.",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Use config only; do not load checkpoint weights.",
    )
    parser.add_argument(
        "--check-output",
        action="store_true",
        help="In compare mode, check compiled logits against eager.",
    )
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.02)
    args = parser.parse_args()
    for name in ("batch_size", "seq_len", "warmup", "steps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not args.model_path and not args.random_init:
        parser.error("--model-path is required unless --random-init is used")
    if args.num_hidden_layers is not None and args.num_hidden_layers <= 0:
        parser.error("--num-hidden-layers must be positive")
    if args.check_output and args.mode != "compare":
        parser.error("--check-output requires --mode compare")
    return args


def main():
    args = parse_args()
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable")
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("This benchmark requires an NPU device")
    torch.npu.set_device(device)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    dtype = getattr(torch, args.dtype)
    config = (
        Qwen25_14BConfig.from_pretrained(args.model_path)
        if args.model_path
        else Qwen25_14BConfig()
    )
    if args.num_hidden_layers is not None:
        if args.num_hidden_layers > config.num_hidden_layers:
            raise ValueError(
                "--num-hidden-layers must not exceed the checkpoint layer count"
            )
        config.num_hidden_layers = args.num_hidden_layers
    if args.seq_len > config.max_position_embeddings:
        raise ValueError("--seq-len exceeds max_position_embeddings")
    if args.random_init:
        model = Qwen2ForCausalLM(config).to(dtype=dtype)
    else:
        model = Qwen2ForCausalLM.from_pretrained(
            args.model_path, config=config, torch_dtype=dtype
        )
    model = model.eval().to(device)
    input_ids = torch.randint(
        0, config.vocab_size, (args.batch_size, args.seq_len), device=device
    )
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "use_cache": False,
    }
    print(
        f"batch_size={args.batch_size}, seq_len={args.seq_len}, dtype={args.dtype}, device={device}",
        flush=True,
    )
    print(
        f"num_hidden_layers={config.num_hidden_layers}, random_init={args.random_init}, seed={args.seed}",
        flush=True,
    )
    modes = ("eager", "autofuse") if args.mode == "compare" else (args.mode,)
    reference = None
    timings = {}
    for mode in modes:
        measured_model = compile_model(model, mode)
        first_ms, avg_ms = benchmark(measured_model, inputs, args.warmup, args.steps)
        timings[mode] = avg_ms
        print(
            f"mode={mode}, first_forward_ms={first_ms:.3f}, avg_forward_ms={avg_ms:.3f}",
            flush=True,
        )
        if args.check_output:
            with torch.inference_mode():
                logits = measured_model(**inputs).logits.float().cpu()
            if reference is None:
                reference = logits
            else:
                max_abs_error = (logits - reference).abs().max().item()
                torch.testing.assert_close(
                    logits, reference, rtol=args.rtol, atol=args.atol
                )
                print(
                    f"output_check=passed, max_abs_error={max_abs_error:.6g}",
                    flush=True,
                )
    if args.mode == "compare":
        print(f"speedup={timings['eager'] / timings['autofuse']:.3f}x")


if __name__ == "__main__":
    main()
