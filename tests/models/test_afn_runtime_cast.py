"""Validate the runtime replacement independently of NPU code generation."""

from unittest.mock import patch

import pytest
import torch

from models.afn_embedding_microbenchmark.compare_autofuse import parse_args, run
from models.afn_embedding_microbenchmark.runtime_cast import (
    convert_element_type_via_to_copy,
    use_to_copy_for_prims,
)


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.float32])
@pytest.mark.parametrize("layout", ["column", "transpose", "empty", "scalar"])
def test_matches_default_primitive(dtype, layout):
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6) - 11.5
    tensor = {
        "column": base[:, 1:2],
        "transpose": base.t(),
        "empty": base[:0],
        "scalar": base[0, 0],
    }[layout]
    primitive = torch.ops.prims.convert_element_type.default
    expected = primitive(tensor, dtype)
    with use_to_copy_for_prims(True, dispatch_key="CPU"):
        actual = primitive(tensor, dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == expected.dtype
    assert actual.stride() == expected.stride()
    assert actual.device == tensor.device
    if tensor.numel():
        assert actual.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()
    assert not torch._C._dispatch_has_kernel_for_dispatch_key(
        "prims::convert_element_type", "CPU"
    )


def test_dispatch_uses_replacement_and_restores_after_exception():
    tensor = torch.ones(3)
    with patch(
        "models.afn_embedding_microbenchmark.runtime_cast.convert_element_type_via_to_copy",
        wraps=convert_element_type_via_to_copy,
    ) as replacement:
        with pytest.raises(ValueError, match="stop"):
            with use_to_copy_for_prims(True, dispatch_key="CPU"):
                torch.ops.prims.convert_element_type.default(tensor, torch.int64)
                replacement.assert_called_once_with(tensor, torch.int64)
                raise ValueError("stop")
        torch.ops.prims.convert_element_type.default(tensor, torch.int64)
        assert replacement.call_count == 1


def test_disabled_override_and_existing_kernel():
    with use_to_copy_for_prims(False, dispatch_key="CPU"):
        assert not torch._C._dispatch_has_kernel_for_dispatch_key(
            "prims::convert_element_type", "CPU"
        )
    with use_to_copy_for_prims(True, dispatch_key="CPU"):
        with pytest.raises(RuntimeError, match="already has a CPU kernel"):
            with use_to_copy_for_prims(True, dispatch_key="CPU"):
                pytest.fail("must not overwrite the active kernel")
        result = torch.ops.prims.convert_element_type.default(torch.ones(1), torch.int64)
        assert result.dtype == torch.int64


def test_switch_requires_npu():
    assert parse_args(["--prims-cast-to-copy"]).prims_cast_to_copy
    with pytest.raises(SystemExit):
        parse_args(["--device", "cpu", "--mode", "eager", "--prims-cast-to-copy"])


def test_override_only_covers_autofuse_timing_and_profile(tmp_path):
    args = parse_args([
        "--device", "cpu", "--mode", "eager", "--vocab-size", "32",
        "--embed-dim", "4", "--check-output",
    ])
    # Exercise orchestration with a CPU dispatcher override instead of an NPU.
    args.mode = "compare"
    args.prims_cast_to_copy = True
    args.profile_dir = str(tmp_path)
    observed = []

    def record_phase(phase):
        observed.append((phase, torch._C._dispatch_has_kernel_for_dispatch_key(
            "prims::convert_element_type", "CPU"
        )))

    def compile_stub(model, mode):
        record_phase("compile")
        return model

    def benchmark_stub(*args):
        record_phase("benchmark")
        return 1.0, 1.0

    def profile_stub(*args):
        record_phase("profile")

    prefix = "models.afn_embedding_microbenchmark.compare_autofuse."
    with (
        patch(prefix + "use_to_copy_for_prims", side_effect=lambda enabled:
              use_to_copy_for_prims(enabled, dispatch_key="CPU")),
        patch(prefix + "compile_model", side_effect=compile_stub),
        patch(prefix + "benchmark", side_effect=benchmark_stub),
        patch(prefix + "collect_profile", side_effect=profile_stub),
    ):
        run(args)
    assert observed == [
        ("compile", False), ("benchmark", False),
        ("compile", True), ("benchmark", True),
        ("compile", False), ("benchmark", False), ("profile", False),
        ("compile", True), ("benchmark", True), ("profile", True),
    ]
    assert not torch._C._dispatch_has_kernel_for_dispatch_key(
        "prims::convert_element_type", "CPU"
    )
