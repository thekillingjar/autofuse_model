# coding=utf-8
"""Run eager vs AutoFuse benchmarks for standard CNN/Transformer/classic models."""

from __future__ import annotations

import argparse
import traceback
from dataclasses import dataclass
from time import perf_counter

import torch

from models.classic_models_standard.compare_autofuse import make_inputs as make_classic_inputs
from models.classic_models_standard.modeling_classic import MODEL_BUILDERS as CLASSIC_BUILDERS
from models.classic_models_standard.modeling_classic import build_model as build_classic_model
from models.cnn_classification_standard.modeling_cnn import ARCH_BUILDERS as CNN_BUILDERS
from models.cnn_classification_standard.modeling_cnn import build_model as build_cnn_model
from models.transformer_standard.modeling_transformers import MODEL_BUILDERS as TRANSFORMER_BUILDERS
from models.transformer_standard.modeling_transformers import build_model as build_transformer_model


@dataclass
class BenchmarkCase:
    group: str
    name: str


def compile_autofuse(model):
    return torch.compile(
        model,
        backend="inductor",
        dynamic=False,
        fullgraph=True,
        options={"npu_backend": "ascendc"},
    )


def build_case(case: BenchmarkCase, batch_size: int, image_size: int, seq_len: int, device: str):
    if case.group == "cnn":
        model = build_cnn_model(case.name).eval().to(torch.bfloat16).to(device)
        inputs = (
            torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=torch.bfloat16),
        )
        return model, inputs
    if case.group == "transformer":
        shape_arg = seq_len if case.name == "bert" else image_size
        model = build_transformer_model(case.name, seq_len_or_image_size=shape_arg).eval().to(torch.bfloat16).to(device)
        if case.name == "bert":
            input_ids = torch.randint(0, 30522, (batch_size, seq_len), dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            inputs = (input_ids, attention_mask)
        else:
            inputs = (
                torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=torch.bfloat16),
            )
        return model, inputs
    if case.group == "classic":
        model = build_classic_model(case.name).eval().to(torch.bfloat16).to(device)
        inputs = make_classic_inputs(case.name, batch_size, device)
        return model, inputs
    raise ValueError(f"Unsupported case group: {case.group}")


def run_model(model, inputs, case: BenchmarkCase):
    if case.group == "transformer" and case.name == "bert":
        return model(inputs[0], attention_mask=inputs[1])
    return model(*inputs)


def time_forward(model, inputs, case: BenchmarkCase, warmup: int, steps: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            run_model(model, inputs, case)
        torch.npu.synchronize()
        start = perf_counter()
        for _ in range(steps):
            run_model(model, inputs, case)
        torch.npu.synchronize()
    return (perf_counter() - start) * 1000 / steps


def run_case(case: BenchmarkCase, args):
    eager_model, eager_inputs = build_case(case, args.batch_size, args.image_size, args.seq_len, "npu")
    eager_ms = time_forward(eager_model, eager_inputs, case, args.warmup, args.steps)
    del eager_model
    torch.npu.empty_cache()

    autofuse_model, autofuse_inputs = build_case(case, args.batch_size, args.image_size, args.seq_len, "npu")
    autofuse_model = compile_autofuse(autofuse_model)
    autofuse_ms = time_forward(autofuse_model, autofuse_inputs, case, args.warmup, args.steps)
    speedup = eager_ms / autofuse_ms if autofuse_ms > 0 else float("inf")
    return eager_ms, autofuse_ms, speedup


def format_row(values: list[str], widths: list[int]) -> str:
    return "  ".join(value.ljust(width) for value, width in zip(values, widths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("cnn", "transformer", "classic"),
        default=("cnn", "transformer", "classic"),
    )
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--show-traceback", action="store_true")
    args = parser.parse_args()

    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable")

    cases: list[BenchmarkCase] = []
    if "cnn" in args.groups:
        cases.extend(BenchmarkCase("cnn", name) for name in sorted(CNN_BUILDERS))
    if "transformer" in args.groups:
        cases.extend(BenchmarkCase("transformer", name) for name in sorted(TRANSFORMER_BUILDERS))
    if "classic" in args.groups:
        cases.extend(BenchmarkCase("classic", name) for name in sorted(CLASSIC_BUILDERS))

    widths = [14, 24, 14, 14, 10, 36]
    print(format_row(["group", "model", "eager_ms", "autofuse_ms", "speedup", "status"], widths))
    print(format_row(["-" * (w - 1) for w in widths], widths))

    for case in cases:
        try:
            eager_ms, autofuse_ms, speedup = run_case(case, args)
            row = [
                case.group,
                case.name,
                f"{eager_ms:.3f}",
                f"{autofuse_ms:.3f}",
                f"{speedup:.2f}x",
                "ok",
            ]
        except Exception as exc:  # noqa: BLE001 - benchmark should report per-model failures.
            if args.show_traceback:
                traceback.print_exc()
            row = [case.group, case.name, "-", "-", "-", f"FAILED: {type(exc).__name__}: {exc}"]
            if not args.continue_on_error:
                print(format_row(row, widths))
                raise
        print(format_row(row, widths))


if __name__ == "__main__":
    main()
