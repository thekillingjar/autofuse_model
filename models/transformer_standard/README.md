# Transformer Standard AutoFuse Comparison

This directory provides standard PyTorch BERT and ViT style models for comparing
eager execution with `torch.compile(..., options={"npu_backend": "ascendc"})`.
The models are randomly initialized and do not depend on Hugging Face weights.

Supported model types:

```text
bert
vit
```

BERT:

```bash
python -m models.transformer_standard.compare_autofuse \
  --model-type bert \
  --mode autofuse \
  --batch-size 1 \
  --seq-len 128
```

ViT:

```bash
python -m models.transformer_standard.compare_autofuse \
  --model-type vit \
  --mode autofuse \
  --batch-size 1 \
  --image-size 224
```

Use `--num-hidden-layers 1` for quick compile smoke tests.

Run all standard CNN/Transformer/classic benchmarks:

```bash
python -m models.benchmark_standard_models \
  --batch-size 1 \
  --image-size 224 \
  --seq-len 128
```
