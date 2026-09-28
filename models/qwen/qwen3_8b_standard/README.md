# Standalone Qwen3-8B AutoFuse baseline

This directory is an independent standard-PyTorch Qwen3-8B implementation.
It does not import the optimized `models/qwen/models/modeling_qwen.py`, the
repository `module/` package, `executor/`, or any `torch_npu.npu_*` operator.

The only experiment variable is the compilation mode:

```text
none       standard PyTorch eager
inductor   torch.compile with the default Inductor lowering
autofuse   torch.compile with options={"npu_backend": "ascendc"}
```

Run from the repository root:

```bash
python3 -m models.qwen.qwen3_8b_standard.compare_autofuse \
    --model-path /data/models/Qwen3-8B \
    --mode none

python3 -m models.qwen.qwen3_8b_standard.compare_autofuse \
    --model-path /data/models/Qwen3-8B \
    --mode inductor

python3 -m models.qwen.qwen3_8b_standard.compare_autofuse \
    --model-path /data/models/Qwen3-8B \
    --mode autofuse
```

The benchmark uses a fixed-shape forward pass and identical random input for
each mode. It intentionally does not benchmark `generate()` because dynamic
cache and sampling control flow would mix with the AutoFuse comparison.
