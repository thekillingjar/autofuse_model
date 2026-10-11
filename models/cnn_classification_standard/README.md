# CNN Classification Standard AutoFuse Comparison

This directory provides standard PyTorch CNN classification models for comparing
eager execution with `torch.compile(..., options={"npu_backend": "ascendc"})`.
The implementations do not depend on torchvision or repository-specific fused
operators.

Supported architectures:

```text
vgg19
resnet50
mobilenet_v2
```

Run from the repository root:

```bash
python -m models.cnn_classification_standard.compare_autofuse \
  --arch resnet50 \
  --mode none \
  --batch-size 1 \
  --image-size 224
```

AutoFuse:

```bash
export AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true;--debug_dir=/tmp/autofuse_debug"
export TORCH_COMPILE_DEBUG=1
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

python -m models.cnn_classification_standard.compare_autofuse \
  --arch resnet50 \
  --mode autofuse \
  --batch-size 1 \
  --image-size 224
```

CNNs are useful AutoFuse sanity tests because their forward graphs are static
and avoid MoE routing control flow.

Run all standard CNN/Transformer/classic benchmarks:

```bash
python -m models.benchmark_standard_models \
  --batch-size 1 \
  --image-size 224 \
  --seq-len 128
```
