"""CPU semantic and harness checks for the isolated AFN lookup reproducer."""

from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from models.afn_embedding_microbenchmark.compare_autofuse import (
    benchmark,
    collect_profile,
    compile_model,
    inspect_profile,
    parse_args,
)
from models.afn_embedding_microbenchmark.modeling_embedding import (
    CASES,
    AFNEmbeddingLookup,
    prepare_input,
)


def make_x():
    return (torch.arange(4 * 39).reshape(4, 39) % 17).float()


@pytest.mark.parametrize("fields", [1, 26])
@pytest.mark.parametrize("case", CASES)
def test_matches_original_expression(fields, case):
    model = AFNEmbeddingLookup(num_fields=fields, vocab_size=17, embed_dim=8, case=case)
    x = make_x()
    actual = model(prepare_input(model, x))
    assert len(actual) == fields
    for result, (name, (start, end)) in zip(
        actual, model.feature_index.items(), strict=True
    ):
        ids = x[:, start:end].long()
        expected = ids if case == "slice_cast" else model.embedding_dict[name](ids)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert result.shape == ((4, 1) if case == "slice_cast" else (4, 1, 8))


def test_ablations_keep_weights_and_output():
    reference = AFNEmbeddingLookup(num_fields=3, vocab_size=17, embed_dim=8)
    x = make_x()
    for case in ("cast_embedding", "embedding_only"):
        model = AFNEmbeddingLookup(num_fields=3, vocab_size=17, embed_dim=8, case=case)
        model.load_state_dict(reference.state_dict(), strict=True)
        for result, expected in zip(
            model(prepare_input(model, x)), reference(x), strict=True
        ):
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
    columns = prepare_input(AFNEmbeddingLookup(case="cast_embedding"), x)
    assert columns[0].stride() == x[:, 13:14].stride()


@pytest.mark.parametrize("case", CASES)
def test_fullgraph_capture(case):
    model = AFNEmbeddingLookup(
        num_fields=2, vocab_size=17, embed_dim=8, case=case
    ).eval()
    inputs = prepare_input(model, make_x())
    compiled = torch.compile(model, backend="eager", dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for actual, expected in zip(compiled(inputs), model(inputs), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_compile_options_and_timing():
    model = MagicMock()
    with patch("torch.compile") as compile_fn:
        compile_model(model, "autofuse")
        compile_fn.assert_called_once_with(
            model,
            backend="inductor",
            dynamic=False,
            fullgraph=True,
            options={"npu_backend": "ascendc"},
        )
    with (
        patch("time.perf_counter", side_effect=[0, 4, 10, 10.02]),
        patch("torch.npu", create=True) as npu,
    ):
        first, avg = benchmark(model, torch.ones(1), SimpleNamespace(type="npu"), 2, 4)
        assert model.call_count == 7
        assert npu.synchronize.call_count == 4
        assert first == 4000
        assert avg == pytest.approx(5)


@pytest.mark.parametrize(
    "argv",
    [
        ["--steps", "0"],
        ["--num-fields", "27"],
        ["--field-offset", "-1"],
        ["--device", "cpu"],
        ["--device", "cpu", "--mode", "eager", "--profile-dir", "prof"],
    ],
)
def test_invalid_cli(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_valid_cli_and_invalid_model():
    args = parse_args(["--device", "cpu", "--mode", "eager", "--num-fields", "26"])
    assert args.batch_size == 128
    with pytest.raises(ValueError):
        AFNEmbeddingLookup(num_fields=27)


def test_profile_schedule_and_metrics(tmp_path):
    fake_module = ModuleType("torch_npu")
    fake_module.profiler = MagicMock()
    api = fake_module.profiler
    model = MagicMock()
    args = SimpleNamespace(
        profile_dir=str(tmp_path),
        case="slice_cast_embedding",
        profile_steps=10,
        with_stack=False,
    )
    with (
        patch.dict("sys.modules", {"torch_npu": fake_module}),
        patch("torch.npu", create=True),
        patch(
            "models.afn_embedding_microbenchmark.compare_autofuse.inspect_profile"
        ) as inspect,
    ):
        collect_profile(model, torch.ones(1), args, "eager")
    api.schedule.assert_called_once_with(wait=0, warmup=2, active=10, repeat=1)
    api._ExperimentalConfig.assert_called_once_with(
        profiler_level=api.ProfilerLevel.Level1,
        aic_metrics=api.AiCMetrics.PipeUtilization,
        export_type=api.ExportType.Text,
    )
    assert model.call_count == 13
    assert api.profile.return_value.__enter__.return_value.step.call_count == 13
    assert api.profile.call_args.kwargs["record_shapes"] is True
    inspect.assert_called_once()


def test_missing_profile_is_reported(tmp_path):
    with pytest.warns(UserWarning, match="No ASCEND_PROFILER_OUTPUT"):
        inspect_profile(tmp_path)
