# Gemma4-31B Standard AutoFuse Comparison

This directory contains a standalone standard-PyTorch Gemma4 text model for
comparing eager execution with `torch.compile(..., options={"npu_backend": "ascendc"})`.
It intentionally avoids repository executor components and torch_npu fused MoE
operators so AutoFuse can see ordinary PyTorch operators.

Example:

```bash
python -m models.gemma_4_31b_standard.compare_autofuse \
  --model-path /path/to/gemma-4-31b \
  --mode none \
  --batch-size 1 \
  --seq-len 128 \
  --num-hidden-layers 1
```

AutoFuse:

```bash
export AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true;--debug_dir=/tmp/autofuse_debug"
export TORCH_COMPILE_DEBUG=1
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

python -m models.gemma_4_31b_standard.compare_autofuse \
  --model-path /path/to/gemma-4-31b \
  --mode autofuse \
  --batch-size 1 \
  --seq-len 128 \
  --num-hidden-layers 1
```

The script reads `config.json` only and constructs a randomly initialized model.
Use `--num-hidden-layers` for compile smoke tests because the full 31B structure
requires substantial memory even without loading checkpoint weights.
