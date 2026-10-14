"""Tiny CPU checks; real NPU AutoFuse compilation is intentionally separate."""

from unittest.mock import patch

import pytest
import torch
from transformers import Qwen2Config as HFQwen2Config
from transformers import Qwen2ForCausalLM as HFQwen2ForCausalLM

from models.qwen2_5_14b_standard.compare_autofuse import benchmark, compile_model
from models.qwen2_5_14b_standard.configuration_qwen2_5_14b import Qwen25_14BConfig
from models.qwen2_5_14b_standard.modeling_qwen2_5_14b import (
    Qwen2ForCausalLM,
    make_causal_mask,
)


def tiny_config(**kwargs):
    settings = {
        "vocab_size": 97,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "use_cache": False,
    }
    settings.update(kwargs)
    return Qwen25_14BConfig(**settings)


def test_checkpoint_defaults():
    config = Qwen25_14BConfig()
    assert (config.num_hidden_layers, config.hidden_size, config.intermediate_size) == (
        48,
        5120,
        13824,
    )
    assert (
        config.num_attention_heads,
        config.num_key_value_heads,
        config.vocab_size,
    ) == (40, 8, 152064)
    assert config.rope_theta == 1000000.0
    assert config.rope_parameters == {"rope_type": "default", "rope_theta": 1000000.0}
    assert not config.use_sliding_window


def test_rope_config_roundtrip(tmp_path):
    config = tiny_config(rope_theta=500000.0)
    config.save_pretrained(tmp_path)
    loaded = Qwen25_14BConfig.from_pretrained(tmp_path)
    assert loaded.rope_theta == config.rope_theta
    assert loaded.rope_parameters == config.rope_parameters
    torch.testing.assert_close(
        Qwen2ForCausalLM(loaded).model.rotary_emb.inv_freq,
        Qwen2ForCausalLM(config).model.rotary_emb.inv_freq,
    )


@pytest.mark.parametrize(
    "mask_kind", ["none", "full", "left_padding", "right_padding", "4d"]
)
def test_huggingface_eager_logits(mask_kind):
    torch.manual_seed(7)
    config = tiny_config()
    model = Qwen2ForCausalLM(config).eval()
    hf_config = HFQwen2Config(**config.to_dict())
    hf_config._attn_implementation = "eager"
    reference = HFQwen2ForCausalLM(hf_config).eval()
    reference.load_state_dict(model.state_dict(), strict=True)
    input_ids = torch.randint(1, config.vocab_size, (2, 6))
    mask = None
    if mask_kind != "none":
        mask = torch.ones_like(input_ids)
        if mask_kind == "left_padding":
            mask[0, :2] = 0
        elif mask_kind == "right_padding":
            mask[0, -2:] = 0
        elif mask_kind == "4d":
            mask = make_causal_mask(torch.zeros(2, 6, config.hidden_size), mask)
    positions = torch.arange(6).unsqueeze(0).expand(2, -1) + 3
    with torch.inference_mode():
        actual = model(input_ids, attention_mask=mask, position_ids=positions).logits
        expected = reference(
            input_ids, attention_mask=mask, position_ids=positions, use_cache=False
        ).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(actual).all()


def test_hf_sharded_checkpoint_load(tmp_path):
    config = tiny_config()
    hf_config = HFQwen2Config(**config.to_dict())
    hf_config._attn_implementation = "eager"
    reference = HFQwen2ForCausalLM(hf_config).eval()
    reference.save_pretrained(tmp_path, max_shard_size="10KB")
    loaded_config = Qwen25_14BConfig.from_pretrained(tmp_path)
    loaded, info = Qwen2ForCausalLM.from_pretrained(
        tmp_path, config=loaded_config, output_loading_info=True
    )
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    assert not info["mismatched_keys"]
    for name, weight in reference.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], weight)
    input_ids = torch.ones(2, 6, dtype=torch.long)
    with torch.inference_mode():
        torch.testing.assert_close(
            loaded(input_ids).logits,
            reference(input_ids, use_cache=False).logits,
            rtol=1e-5,
            atol=1e-6,
        )


def test_bfloat16_and_rope_precision():
    model = Qwen2ForCausalLM(tiny_config()).eval()
    inv_freq = model.model.rotary_emb.inv_freq.clone()
    model = model.to(torch.bfloat16)
    torch.testing.assert_close(
        model.model.rotary_emb.inv_freq, inv_freq, rtol=0, atol=0
    )
    with torch.inference_mode():
        logits = model(torch.ones(2, 6, dtype=torch.long)).logits
    assert logits.dtype == torch.bfloat16
    assert logits.shape == (2, 6, 97)
    assert torch.isfinite(logits).all()


def test_fullgraph_capture():
    model = Qwen2ForCausalLM(tiny_config()).eval()
    compiled = torch.compile(model, backend="eager", dynamic=False, fullgraph=True)
    input_ids = torch.ones(2, 6, dtype=torch.long)
    with torch.inference_mode():
        torch.testing.assert_close(compiled(input_ids).logits, model(input_ids).logits)


def test_input_embeddings_and_tuple_outputs():
    model = Qwen2ForCausalLM(tiny_config()).eval()
    input_ids = torch.ones(2, 6, dtype=torch.long)
    with torch.inference_mode():
        embeddings = model.get_input_embeddings()(input_ids)
        actual = model(inputs_embeds=embeddings, return_dict=False)[0]
        torch.testing.assert_close(actual, model(input_ids).logits)


def test_unsupported_settings_fail():
    with pytest.raises(ValueError, match="sliding"):
        Qwen2ForCausalLM(tiny_config(use_sliding_window=True))
    with pytest.raises(ValueError, match="RoPE"):
        Qwen2ForCausalLM(
            tiny_config(rope_scaling={"rope_type": "linear", "factor": 2.0})
        )
    model = Qwen2ForCausalLM(tiny_config())
    with pytest.raises(ValueError, match="use_cache"):
        model(torch.ones(1, 4, dtype=torch.long), use_cache=True)


def test_compile_modes():
    model = object()
    assert compile_model(model, "none") is model
    assert compile_model(model, "eager") is model
    with patch("torch.compile", return_value="compiled") as compiler:
        assert compile_model(model, "autofuse") == "compiled"
        compiler.assert_called_once_with(
            model,
            backend="inductor",
            dynamic=False,
            fullgraph=True,
            options={"npu_backend": "ascendc"},
        )


def test_benchmark_excludes_first_forward_and_warmup():
    with (
        patch("torch.npu", create=True) as npu,
        patch("time.perf_counter", side_effect=[0, 5, 10, 10.02]),
    ):
        with patch("torch.nn.Identity.forward") as forward:
            first_ms, avg_ms = benchmark(
                torch.nn.Identity(), {"input": torch.ones(1)}, warmup=2, steps=4
            )
        assert forward.call_count == 7
        assert npu.synchronize.call_count == 4
        assert first_ms == 5000
        assert avg_ms == pytest.approx(5)
