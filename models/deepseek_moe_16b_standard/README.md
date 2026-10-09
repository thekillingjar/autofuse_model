# DeepSeek-MoE 16B Standard AutoFuse Comparison

This directory provides a standalone PyTorch reference for
`deepseek-ai/deepseek-moe-16b-base` eager, Inductor, and AutoFuse comparison.
It does not use the repository `module/` package, `executor/`, fused attention,
or `torch_npu.npu_*` operators.

Modes:

```text
none       eager PyTorch execution
inductor   torch.compile with the default Inductor path
autofuse   torch.compile with options={"npu_backend": "ascendc"}
```

Run:

```bash
python3 -m models.deepseek_moe_16b_standard.compare_autofuse \
  --model-path /path/to/deepseek-moe-16b-base \
  --mode none
```

```bash
python3 -m models.deepseek_moe_16b_standard.compare_autofuse \
  --model-path /path/to/deepseek-moe-16b-base \
  --mode autofuse
```

For compile smoke tests, cap the layer count:

```bash
python3 -m models.deepseek_moe_16b_standard.compare_autofuse \
  --model-path /path/to/deepseek-moe-16b-base \
  --mode autofuse \
  --num-hidden-layers 1 \
  --seq-len 16 \
  --warmup 1 \
  --steps 1
```

The MoE block uses a fixed-graph PyTorch implementation that computes every
routed expert and masks by top-k router weights. This keeps fullgraph compile
simple and avoids custom sparse MoE kernels, but it is not a production sparse
MoE performance implementation.
