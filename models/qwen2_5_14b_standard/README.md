# Qwen2.5-14B Eager / AutoFuse Comparison

Standalone, no-cache PyTorch implementation of the Hugging Face
`Qwen/Qwen2.5-14B` base model. Configuration and checkpoint parameter names
follow Qwen2: 48 layers, hidden size 5120, intermediate size 13824, 40 query
heads, 8 KV heads, and vocabulary size 152064.

The modeling file implements RMSNorm, FP32 RoPE, GQA, explicit attention
matmul/softmax, and SwiGLU with standard PyTorch operations. It does not import
other repository models, NPU fused operators, SDPA, or FlashAttention.
Configuration and modeling are adapted from the Apache-2.0 Qwen2 implementation;
the upstream notices are retained in the source headers. The existing repository
`Third_Party_Open_Source_Software_Notice` contains the Transformers license notice.

Sources:

- https://huggingface.co/Qwen/Qwen2.5-14B/blob/main/config.json
- https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2/modeling_qwen2.py
- https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2/configuration_qwen2.py

## Environment

Install mutually compatible PyTorch, torch_npu, CANN, and AutoFuse versions
on the target NPU machine. The `ascendc` option requires an environment that
provides that Inductor backend; merely installing upstream PyTorch is insufficient.
This experimental AutoFuse benchmark is separate from the repository's
executor-based GE / npugraph_ex inference paths.

From the repository root, enter the model directory before running:

```bash
cd models/qwen2_5_14b_standard
export PYTHONPATH="$PWD/../..:${PYTHONPATH:-}"
python -m pip install -r requirements.txt
```

Download the model weights separately and replace `/path/to/Qwen2.5-14B`
below with the actual checkpoint directory. The benchmark does not need a
tokenizer because it creates seeded random token IDs.

## Compare Both Modes

```bash
python -m models.qwen2_5_14b_standard.compare_autofuse \
  --model-path /path/to/Qwen2.5-14B \
  --mode compare \
  --batch-size 1 \
  --seq-len 128 \
  --warmup 5 \
  --steps 20 \
  --check-output
```

Both modes use the same model weights, BF16 dtype, input IDs, attention mask,
and `use_cache=False`. The optional output check is outside the timed region;
adjust `--rtol` and `--atol` if a different numerical tolerance is required.
It is a tolerance check, not a proof of accuracy for all prompts.

`first_forward_ms` includes cold initialization / compilation. The additional
warmup iterations and first forward are excluded from `avg_forward_ms`.
NPU synchronization brackets the timed loop; the average is wall-clock forward
latency including host dispatch, not a sum of device kernel durations.
`speedup = eager_ms / autofuse_ms`; values greater than 1 indicate improvement.

## Separate Processes

```bash
python -m models.qwen2_5_14b_standard.compare_autofuse \
  --model-path /path/to/Qwen2.5-14B --mode none \
  --batch-size 1 --seq-len 128 --warmup 5 --steps 20

python -m models.qwen2_5_14b_standard.compare_autofuse \
  --model-path /path/to/Qwen2.5-14B --mode autofuse \
  --batch-size 1 --seq-len 128 --warmup 5 --steps 20
```

`--mode eager` aliases `none`. `--mode inductor` leaves backend options empty
and is only a diagnostic path if your environment supports it. AutoFuse uses:

```python
model = torch.compile(
    model,
    backend="inductor",
    dynamic=False,
    fullgraph=True,
    options={"npu_backend": "ascendc"},
)
```

No compilation exception is suppressed and no automatic eager fallback is
installed. Selecting this option does not prove every operation was fused;
inspect the generated code / profiler to identify actual kernel dispatch.

## Smoke Test

```bash
python -m models.qwen2_5_14b_standard.compare_autofuse \
  --random-init --num-hidden-layers 1 --mode compare \
  --batch-size 1 --seq-len 16 --warmup 2 --steps 5 --check-output
```

This still allocates the full-sized embedding and LM head. Layer-capped or
random-initialized runs are not full checkpoint / language-quality benchmarks.

## Scope and Tests

The benchmark measures a fixed-shape prefill-style full forward, including
logits for every token. It is not autoregressive decoding or generation.
KV cache, training loss, scaled RoPE, and sliding-window attention are not
implemented; unsupported cache / RoPE / sliding-window settings fail explicitly.
Attention materializes quadratic score tensors; large batch / sequence lengths
can exhaust memory. No hardware capacity or performance is assumed here.

CPU tests use a tiny configuration, match weights and logits against HF eager
Qwen2, and test masks, checkpoint loading, and full-graph capture:

```bash
python -m pytest ../../tests/models/test_qwen2_5_14b_standard.py -q
```

Actual NPU compilation and timings must be verified on the target machine.
The tiny CPU suite passed on PyTorch 2.8.0 with both Transformers 4.45.2
and 5.0.0. This does not validate an NPU compiler build or resolve existing
Inductor backend issues in the target environment.
