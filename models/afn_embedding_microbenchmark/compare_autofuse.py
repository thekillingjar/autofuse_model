"""Measure AFN sparse lookup latency without the rest of the AFN network."""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import torch

from .modeling_embedding import CASES, AFNEmbeddingLookup, prepare_input


def compile_model(model, mode):
    if mode in ("none", "eager"):
        return model
    if mode not in ("autofuse", "inductor"):
        raise ValueError(f"Unknown mode: {mode}")
    options = {"npu_backend": "ascendc"} if mode == "autofuse" else {}
    return torch.compile(
        model,
        backend="inductor",
        dynamic=False,
        fullgraph=True,
        options=options,
    )


def synchronize(device):
    if device.type == "npu":
        torch.npu.synchronize()


@torch.inference_mode()
def benchmark(model, inputs, device, warmup, steps):
    synchronize(device)
    start = time.perf_counter()
    model(inputs)
    synchronize(device)
    first_ms = (time.perf_counter() - start) * 1000
    for _ in range(warmup):
        model(inputs)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(steps):
        model(inputs)
    synchronize(device)
    return first_ms, (time.perf_counter() - start) * 1000 / steps


def inspect_profile(directory):
    """Report export problems instead of silently accepting an empty profile."""
    outputs = list(directory.rglob("ASCEND_PROFILER_OUTPUT"))
    if not outputs:
        warnings.warn(
            f"No ASCEND_PROFILER_OUTPUT found under {directory}", stacklevel=2
        )
    for output in outputs:
        print(f"profile_output={output}", flush=True)
        for name in (
            "kernel_details.csv",
            "op_statistic.csv",
            "api_statistic.csv",
            "step_trace_time.csv",
        ):
            path = output / name
            if not path.exists():
                warnings.warn(f"Missing profiler artifact: {path}", stacklevel=2)
                continue
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
                has_rows = next(reader, None) is not None
            print(f"{name}: columns={len(header)}, has_rows={has_rows}", flush=True)
            if not has_rows:
                warnings.warn(f"Empty profiler table: {path}", stacklevel=2)
        trace = output / "trace_view.json"
        if not trace.exists():
            warnings.warn(f"Missing profiler artifact: {trace}", stacklevel=2)
        else:
            with trace.open(encoding="utf-8") as handle:
                json.load(handle)
            print("trace_view.json: valid", flush=True)


@torch.inference_mode()
def collect_profile(model, inputs, args, mode):
    import torch_npu

    directory = (
        Path(args.profile_dir).resolve() / f"{args.case}_{mode}_{time.time_ns()}"
    )
    directory.mkdir(parents=True)
    experimental = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        export_type=torch_npu.profiler.ExportType.Text,
    )
    # Compilation already happened in benchmark(); profile only steady-state calls.
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=2, active=args.profile_steps, repeat=1
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory)),
        record_shapes=True,
        with_stack=args.with_stack,
        experimental_config=experimental,
    ) as prof:
        for _ in range(2 + args.profile_steps + 1):
            with torch.profiler.record_function("afn_sparse_lookup"):
                model(inputs)
            prof.step()
        torch.npu.synchronize()
    inspect_profile(directory)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("none", "eager", "autofuse", "inductor", "compare"),
        default="compare",
    )
    parser.add_argument("--case", choices=CASES, default="slice_cast_embedding")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-fields", type=int, default=1)
    parser.add_argument("--total-fields", type=int, default=39)
    parser.add_argument("--field-offset", type=int, default=13)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument(
        "--index-range",
        type=int,
        default=32,
        help="Sample IDs in [0, min(index_range, vocab_size)).",
    )
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
        help="Embedding weight dtype; X remains FP32.",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--check-output", action="store_true")
    parser.add_argument(
        "--profile-dir",
        help="Optionally collect an additional NPU profile, outside latency timing.",
    )
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument(
        "--with-stack",
        action="store_true",
        help="Collect extra CPU stacks; increases profiling overhead.",
    )
    args = parser.parse_args(argv)
    for name in (
        "batch_size",
        "num_fields",
        "total_fields",
        "vocab_size",
        "index_range",
        "embed_dim",
        "warmup",
        "steps",
        "profile_steps",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.field_offset < 0 or args.field_offset + args.num_fields > args.total_fields:
        parser.error("field-offset + num-fields must fit inside total-fields")
    if args.profile_dir and not args.device.startswith("npu"):
        parser.error("--profile-dir requires an NPU device")
    if args.device.startswith("cpu") and args.mode not in ("none", "eager"):
        parser.error("CPU smoke tests require --mode eager or none")
    return args


def main():
    args = parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError("torch.npu is unavailable")
        torch.npu.set_device(args.device)
    device = torch.device(args.device)
    if device.type not in ("cpu", "npu"):
        raise ValueError("Use an NPU device, or CPU for an eager smoke test")
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    model = (
        AFNEmbeddingLookup(
            num_fields=args.num_fields,
            total_fields=args.total_fields,
            field_offset=args.field_offset,
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            case=args.case,
        )
        .eval()
        .requires_grad_(False)
        .to(device=device, dtype=dtype)
    )
    # Build valid categorical IDs; unlike the current AFN wrapper, never cast rand([0,1)) to IDs.
    x = (
        torch.randint(
            0,
            min(args.index_range, args.vocab_size),
            (args.batch_size, args.total_fields),
        )
        .float()
        .to(device)
    )
    inputs = prepare_input(model, x)
    synchronize(device)
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(
        f"case={args.case}, batch_size={args.batch_size}, num_fields={args.num_fields}, embed_dim={args.embed_dim}",
        flush=True,
    )
    print(
        f"device={device}, weight_dtype={args.dtype}, input_dtype={x.dtype}, input_stride={x.stride()}",
        flush=True,
    )
    print(
        f"vocab_size={args.vocab_size}, sampled_id_range={min(args.index_range, args.vocab_size)}, parameter_bytes={parameter_bytes}",
        flush=True,
    )
    modes = ("eager", "autofuse") if args.mode == "compare" else (args.mode,)
    reference = None
    if args.check_output:
        with torch.inference_mode():
            reference = tuple(value.cpu().clone() for value in model(inputs))
    timings = {}
    for mode in modes:
        measured_model = compile_model(model, mode)
        first_ms, avg_ms = benchmark(
            measured_model, inputs, device, args.warmup, args.steps
        )
        timings[mode] = avg_ms
        print(
            f"mode={mode}, first_forward_ms={first_ms:.6f}, avg_forward_ms={avg_ms:.6f}",
            flush=True,
        )
        if reference is not None:
            with torch.inference_mode():
                outputs = measured_model(inputs)
                for actual, expected in zip(outputs, reference, strict=True):
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
            print(f"mode={mode}, output_check=passed", flush=True)
        del measured_model
    if args.mode == "compare":
        print(f"speedup={timings['eager'] / timings['autofuse']:.3f}x", flush=True)
    # Finish all unprofiled timings first so profiling eager cannot perturb AF timing.
    if args.profile_dir:
        for mode in modes:
            measured_model = compile_model(model, mode)
            benchmark(measured_model, inputs, device, args.warmup, 1)
            collect_profile(measured_model, inputs, args, mode)


if __name__ == "__main__":
    main()
