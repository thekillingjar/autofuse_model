# Classic Models Standard AutoFuse Comparison

This directory provides standard PyTorch versions of several classic models for
comparing eager execution with `torch.compile(..., options={"npu_backend": "ascendc"})`.

Supported model types:

```text
sam
dien
afn
esmm
decision_transformer
```

Examples:

```bash
python -m models.classic_models_standard.compare_autofuse \
  --model-type afn \
  --mode autofuse \
  --batch-size 128
```

```bash
python -m models.classic_models_standard.compare_autofuse \
  --model-type sam \
  --mode autofuse \
  --batch-size 1
```

The implementations are randomly initialized and use synthetic fixed-shape
inputs. They are intended for AutoFuse operator coverage experiments rather than
task-quality evaluation.

Run all standard CNN/Transformer/classic benchmarks:

```bash
python -m models.benchmark_standard_models \
  --batch-size 1 \
  --image-size 224 \
  --seq-len 128
```
