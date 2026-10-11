# DeepSeek-R1-Distill-Qwen-14B Standard AutoFuse Comparison

This directory contains a standalone Qwen2-style PyTorch implementation for
DeepSeek-R1-Distill-Qwen-14B. It avoids repository executor components and
torch_npu fused operators so eager and AutoFuse paths can be compared on the
same operator-level model.

Run from the repository root:

```bash
python -m models.deepseek_r1_distill_qwen_14b_standard.compare_autofuse \
  --model-path /path/to/DeepSeek-R1-Distill-Qwen-14B \
  --mode none \
  --batch-size 1 \
  --seq-len 128
```

AutoFuse:

```bash
export AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true;--debug_dir=/tmp/autofuse_debug"
export TORCH_COMPILE_DEBUG=1
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

python -m models.deepseek_r1_distill_qwen_14b_standard.compare_autofuse \
  --model-path /path/to/DeepSeek-R1-Distill-Qwen-14B \
  --mode autofuse \
  --batch-size 1 \
  --seq-len 128
```

Use `--random-init --num-hidden-layers 1` for compile smoke tests without
loading checkpoint weights.
