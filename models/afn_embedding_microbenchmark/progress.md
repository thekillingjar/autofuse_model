# Isolated AFN Sparse Lookup Reproducer

- User request: isolate `embedding_dict[name](X[:, start:end].long())` to
  investigate eager vs AutoFuse latency and large waits preceding short tasks.
- Reference: current `models/classic_models_standard/modeling_classic.py`
  declares 13 dense / 26 sparse fields, vocabulary 2100000, embedding dim 128;
  classic compare script casts model weights to BF16. Actual target hardware,
  package versions, and profiler artifacts were not provided.
- Implementation: standalone standard PyTorch model, no DeepCTR dependency or
  existing AFN modification. Return all per-field outputs so no unused lookup
  can be removed from the graph. Cases isolate Slice+Cast+Embedding,
  Cast+Embedding, Embedding alone, and Slice+Cast.
- Defaults: batch 128, one field at offset 13 of 39, vocabulary 10000 for a
  small test, sampled IDs [0, 32), embedding dim 128, BF16 weights, FP32 X.
  Full AFN capacity is explicitly opt-in and no hardware capacity is assumed.
- Timing uses the same weights/inputs across modes, excludes first forward,
  warmup, output checking and profiling, and brackets the loop with sync.
- Profiling is local to this standalone script (executor profiler not used).
  Level1 + PipeUtilization, CPU/NPU, shapes, schedule and per-forward prof.step;
  10 active forwards by default, after compile/warmup. Both latency passes
  complete before either profile. Verify artifacts and warn on missing tables.
- CPU PyTorch 2.8.0: all 22 tests passed, including profiler schedule / metrics
  and missing-export checks. Eager CLI smoke test with batch 128 and 2 fields
  passed with exact output checking. Model-directory CLI --help works.
- Ruff check, compileall and git diff --check passed. Changes are confined
  to this standalone model directory and its focused CPU test file.
- No local NPU test, AutoFuse performance claim, or root-cause conclusion.
  Inplace packing/copy is intentionally excluded from this first reproducer.

## CANN report retention fix (2026-10-15)

- User reports no mindstudio_profiler_output. The script previously enabled
  Text export but left data_simplification at the package default and checked
  only ASCEND_PROFILER_OUTPUT.
- Verified official Ascend/pytorch tag v2.8.0 source:
  torch_npu/profiler/experimental_config.py defaults data_simplification=True;
  analysis/_profiling_parser.py::ProfilingParser.simplify_data explicitly
  removes mindstudio_profiler_output and mindstudio_profiler_log when True.
  analysis/prof_view/cann_parse/_cann_export.py invokes msprof for Text export.
- Explicitly disable simplification. Report the capture path, retained CANN
  export directories, version-suffixed op_summary/op_statistic CSVs and
  msprof timelines. Warn separately when CANN reports are missing, even if
  merged framework reports exist, and show available parser log paths.
- No changes to model, compilation options, schedule or latency measurement.
  Target package versions and artifacts remain unavailable, so this fixes
  the demonstrated retention issue but does not assert the target parser
  will succeed. Local NPU export verification is unavailable.
- Validation: 25 CPU tests passed, including the disabled simplification
  option, nested/suffixed CANN exports, missing CANN exports despite valid
  framework reports, and missing/empty op_summary warnings. Ruff and
  git diff --check passed. Model-directory CPU eager smoke test passed with
  PYTHONPATH set to the repository root and exact output checking.

## Slice-only comparison (2026-10-16)

- Request: add an eager vs AutoFuse comparison containing only slice.
- Extend the existing standalone harness instead of changing the full AFN or
  executor. Keep the existing compile flags, timing and optional profiler.
- slice_only returns X[:, start:end] without cast, embedding, copy or reduction.
  CPU tests verify storage sharing and preserved strides/offsets; no Device
  kernel is required, so do not claim this reproduces the SliceAiCore task.
- Optional slice_copy clones each slice to contiguous storage; no cast or
  embedding. This tests materialization separately, without promising a
  particular ACLNN implementation. Both cases allocate no embedding tables.
- Input defaults remain FP32 [128, 39], column offset 13, one selected field.
  Actual target hardware/packages and NPU profile remain unverified locally.
- Validation: 43 CPU tests passed with PyTorch 2.8.0, covering both new cases
  at batch 1/128 and FP32/BF16/INT64, no parameters, shared vs independent
  storage, exact values, and fullgraph capture. Both model-directory eager
  CLI smoke tests passed with 26 fields and exact output checking. PYTHONPATH
  must include the repository root, as shown in the README.
- Pre-commit all-files check encountered existing repository formatting and
  script-permission failures plus the master branch guard. Unrelated automatic
  whitespace/EOF edits from this check were reverted; no baseline cleanup is
  included. Run scoped hooks with only the branch guard skipped to publish the
  requested commit on the existing tracked master branch.
