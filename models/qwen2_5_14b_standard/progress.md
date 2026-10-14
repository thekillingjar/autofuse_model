# Qwen2.5-14B Comparison Implementation

- Deployment: standalone benchmark, matching the existing standard model directories.
- Source: Qwen/Qwen2.5-14B HF config.json (retrieved through hf-mirror.com);
  HF Transformers v4.45.2 Qwen2 modeling and configuration (GitHub raw source).
- Confirmed config: 48 layers, hidden 5120, intermediate 13824, Q heads 40,
  KV heads 8, vocab 152064, unscaled RoPE theta 1000000, no sliding windows.
- Scope: standard PyTorch no-cache fixed-shape forward, explicit eager attention;
  compare the same weights and inputs using Inductor's ascendc option.
- Local system Python has no torch, torch_npu, transformers, or pytest.
  Temporary CPU test environments were created using uv outside the repository.
- Target NPU hardware, checkpoint path, and deployed compiler versions are not
  supplied. No real checkpoint or NPU performance baseline has been collected.
- Compatibility findings: Transformers 5 normalizes RoPE into rope_parameters;
  explicitly preserve checkpoint theta. Its initialization helpers honor loaded
  parameter flags, preventing checkpoint weights from being reinitialized.
  Non-persistent RoPE buffers are rebuilt for meta checkpoint loading.
- Validation: PyTorch 2.8.0 + Transformers 4.45.2: 14 CPU tests passed;
  PyTorch 2.8.0 + Transformers 5.0.0: 14 CPU tests passed. Includes HF eager
  logits, padding masks, sharded checkpoint weights and logits, BF16 frequency
  preservation, fullgraph capture with CPU eager backend, backend options,
  and synchronization / timing exclusions. Ruff check and compileall passed.
- CLI --help succeeds from the model directory with repository PYTHONPATH.
- Pre-commit: all hooks applicable to the 8 added files passed. Full-repository
  checks were run in an isolated index snapshot and failed on pre-existing
  formatting / script permission issues; unrelated worktree files were untouched.
- No NPU AutoFuse compilation, real 14B checkpoint run, or speedup result is
  claimed. The existing target compiler's reduction issues may still occur.
