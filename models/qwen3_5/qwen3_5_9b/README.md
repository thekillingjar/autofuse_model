# Standalone Qwen3.5-9B AutoFuse baseline

This directory is an independent Dense Qwen3.5 text implementation. It does
not import the existing `models/qwen3_5` MoE model, `module/`, `executor/`, or
any `torch_npu.npu_*` fused operation.

The model uses ordinary PyTorch modules:

```text
nn.Linear
nn.Conv1d
matmul
softmax
RMSNorm
SwiGLU
Gated Delta Rule
```

The only experimental switch is the compile backend:

```python
torch.compile(
    model,
    backend="inductor",
    dynamic=False,
    fullgraph=True,
    options={"npu_backend": "ascendc"},
)
```

Run from the repository root so the relative import works:

```bash
python3 -m models.qwen3_5.qwen3_5_9b.compare_autofuse \
  --model-path /data/models/Qwen3.5-9B \
  --mode none

python3 -m models.qwen3_5.qwen3_5_9b.compare_autofuse \
  --model-path /data/models/Qwen3.5-9B \
  --mode inductor

python3 -m models.qwen3_5.qwen3_5_9b.compare_autofuse \
  --model-path /data/models/Qwen3.5-9B \
  --mode autofuse
```

Use identical `batch_size`, `seq_len`, warmup, and steps. The benchmark uses a
fixed-shape forward pass so Dynamo/Inductor graph changes do not get mixed with
generation control flow.
