# AFN Slice / Cast / Embedding Microbenchmark

Isolate this expression without importing DeepCTR or the full AFN:

```python
embedding_dict[name](X[:, start:end].long())
```

For embedding cases, one independent table is allocated per selected field. The
forward returns a tuple of per-field results, without concatenation, reduction,
input packing, slice assignments, or downstream AFN layers. In particular,
this does NOT reproduce `X_mat[:, ...] = ...` / inplace copy. A missing
`InplaceCopy` kernel is expected and does not itself diagnose the original model.

## Run

Enter this directory from the repository root on the target NPU machine:

```bash
cd models/afn_embedding_microbenchmark
export PYTHONPATH="$PWD/../..:${PYTHONPATH:-}"
```

Use your existing compatible torch / torch_npu / CANN / AutoFuse installation.
No DeepCTR, Transformers, model weights, or datasets are needed.

Start with one field, batch size 128:

```bash
python -m models.afn_embedding_microbenchmark.compare_autofuse \
  --mode compare --batch-size 128 --num-fields 1 \
  --embed-dim 128 --vocab-size 10000 --check-output
```

Reproduce 26 per-field lookups (the deep embedding branch only):

```bash
python -m models.afn_embedding_microbenchmark.compare_autofuse \
  --mode compare --batch-size 128 --num-fields 26 \
  --embed-dim 128 --vocab-size 10000 --check-output
```

Default X is FP32 `[128, 39]`, and selected columns begin at offset 13.
Embedding weights default to BF16, matching the current classic benchmark's
model cast. Set `--dtype float32` if that is the actual AFN run's weight dtype;
use `--embed-dim 1` to test the linear embedding branch separately.

The small default vocabulary saves memory and is NOT the full AFN allocation.
Use `--vocab-size 2100000` to match the existing AFN table capacity if memory
permits. The program prints actual parameter bytes before timing. All selected
fields have separate tables, so increasing field count multiplies that memory.

IDs default to `[0, 32)` to match the existing input generator, but are valid
integer-valued FP32 numbers (not random floats truncated to zero). Set
`--index-range 2100000` along with the full vocabulary to investigate less-local
access patterns. Reusing identical inputs deliberately measures a hot/repeated
lookup scenario; conclusions need not apply to changing IDs or the full AFN.

## Ablations

Choose one `--case` per run, retaining the other parameters:

| Case | Operations inside forward | Preparation outside timing |
|---|---|---|
| `slice_cast_embedding` (default) | `X[:, start:end].long()` then embedding | None |
| `cast_embedding` | `.long()` then embedding | Pre-slice views; preserve their strides |
| `embedding_only` | Embedding only | Pre-slice and convert to INT64 |
| `slice_cast` | Slice and INT64 cast, no table | None |
| `slice_only` | `X[:, start:end]` views only, no cast or table | None |
| `slice_copy` | Slice then `clone(memory_format=torch.contiguous_format)` | None |

```bash
python -m models.afn_embedding_microbenchmark.compare_autofuse \
  --mode compare --case cast_embedding --num-fields 26 --check-output

python -m models.afn_embedding_microbenchmark.compare_autofuse \
  --mode compare --case embedding_only --num-fields 26 --check-output

python -m models.afn_embedding_microbenchmark.compare_autofuse \
  --mode compare --case slice_cast --num-fields 26 --check-output
```

### Slice Only

Compare the exact `X[:, start:end]` expression, with no Cast, Embedding,
concatenation, reduction or inplace assignment:

```bash
python -m models.afn_embedding_microbenchmark.compare_autofuse \
  --mode compare --case slice_only --batch-size 128 --num-fields 1 \
  --check-output --profile-dir ./prof_slice --profile-steps 10
```

The input remains FP32 `[128, 39]`, selecting column 13 by default. There
are no model parameters for either slice case; `--dtype` only applies to
embedding weights and does not change the input dtype. `--num-fields 26`
returns 26 separate slices, without combining them.

Basic slicing returns a view sharing input storage. This case can therefore
run without a Device kernel; its latency primarily measures Host view creation
and dispatch, not the duration of a `SliceAiCore_Slice` kernel. An empty Device
profile or missing AutoFuse kernel is not, on its own, an error.

To force a separate contiguous result for every slice, use the optional copy
case. It still contains no Cast or Embedding, but is **not** a pure view test:

```bash
python -m models.afn_embedding_microbenchmark.compare_autofuse \
  --mode compare --case slice_copy --batch-size 128 --num-fields 1 \
  --check-output --profile-dir ./prof_slice_copy --profile-steps 10
```

`clone` copies even when the selected view is already contiguous (e.g. batch 1).
The backend chooses the actual copy kernel; this does not guarantee an
`aclnnInplaceCopy_SlicAiCore_Slice` kernel or reproduce AFN's inplace packing.
Inspect the generated profile rather than inferring kernel names from the case.

Within compare mode, eager and AutoFuse share the same weights and inputs.
AutoFuse uses `backend="inductor"`, `dynamic=False`, `fullgraph=True`, and
`options={"npu_backend": "ascendc"}`. No errors are suppressed or eager fallback
installed. `--mode none` / `eager` and `--mode autofuse` allow separate processes.
Compilation is not evidence that the backend actually fused every operation.

`first_forward_ms` is reported separately. `avg_forward_ms` excludes the first
forward, warmup, output checks, and profiling. Synchronization brackets the
whole timed loop, not each iteration; results include host dispatch and device
execution. Do not subtract ablation timings as exact individual operator costs:
removing an operation can also change fusion, layouts, and dispatch.

## Profile

```bash
python -m models.afn_embedding_microbenchmark.compare_autofuse \
  --mode compare --num-fields 26 --check-output \
  --profile-dir ./prof_lookup --profile-steps 10
```

Unprofiled timings for both modes finish BEFORE either profiling pass.
CPU + NPU profiling uses Level1 + PipeUtilization, record_shapes=True, an
explicit schedule, and one `prof.step()` per forward. Compilation and warmup
finish before capture. Eager and AF traces are exported to separate directories.
Text export and `data_simplification=False` explicitly retain CANN's parsed
reports and parser logs, in addition to the framework's merged reports. This
uses more disk space than the default simplified export.
`--with-stack` adds CPU call stacks, with extra profiling overhead. Do not wrap
this command in an external msprof collection at the same time.

The script prints `profile_dir` before capture and `cann_profile_output` after
parsing. Typical output layout (worker names and filenames vary by version):

```text
prof_lookup/slice_cast_embedding_<mode>_<timestamp>/
  <worker>_ascend_pt/
    ASCEND_PROFILER_OUTPUT/
      kernel_details.csv
      trace_view.json
    PROF_<id>/
      mindstudio_profiler_output/
        op_summary*.csv
        op_statistic*.csv
        msprof*.json
```

`ASCEND_PROFILER_OUTPUT` is not `mindstudio_profiler_output`. The latter is
CANN's export containing `op_summary*.csv` for the Task Wait Time analysis.
In the official Ascend/pytorch v2.8.0 source, `torch_npu/profiler/experimental_config.py`
defaults `data_simplification=True`, and
`torch_npu/profiler/analysis/_profiling_parser.py::ProfilingParser.simplify_data`
then removes `mindstudio_profiler_output` after analysis. This explains why
the old script could retain framework reports without retaining CANN reports.

The script checks both directories, reports CSV column counts and whether
tables contain data, and checks that trace_view.json parses. Missing CANN
exports or op_summary tables produce explicit warnings. Re-run capture after
updating; the change does not restore previously deleted files.

If CANN reports are still missing, inspect the printed `profile_log` paths and
check `command -v msprof` in the same environment. The official
`torch_npu/profiler/analysis/prof_view/cann_parse/_cann_export.py::CANNExportParser`
uses `msprof --export=on --output=<PROF directory>` for Text export; keeping
data does not fix an unavailable msprof or a failed parser. Do not assume a
missing directory means collection itself failed.

Compare the Slice/Cast/Embedding host dispatch, device durations, stream gaps,
and per-forward task count. If this minimal case is slow under AF, it narrows
the reproducer to this execution path, but does NOT by itself prove an AutoFuse
codegen bug. If it is fast, investigate the original AFN's input packing,
inplace copies, other layers, and their dependencies next.

## Local Checks

```bash
python -m models.afn_embedding_microbenchmark.compare_autofuse \
  --device cpu --mode eager --vocab-size 64 --embed-dim 8 \
  --num-fields 2 --warmup 2 --steps 5 --check-output

python -m pytest ../../tests/models/test_afn_embedding_microbenchmark.py -q
```

CPU checks validate semantics and graph capture only, not NPU compilation,
profile exports, or NPU performance. Target hardware is not assumed.
