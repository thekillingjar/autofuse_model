# coding=utf-8
"""Standalone Qwen3-8B configuration for the AutoFuse experiment."""

from transformers.configuration_utils import PretrainedConfig


class Qwen3Config(PretrainedConfig):
    model_type = "qwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=40960,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_parameters=None,
        partial_rotary_factor=1.0,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=128,
        **kwargs,
    ):
        if rope_parameters is None:
            rope_parameters = {
                "rope_type": "default",
                "rope_theta": rope_theta,
                "partial_rotary_factor": partial_rotary_factor,
            }
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_parameters = rope_parameters
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        super().__init__(
            pad_token_id=kwargs.pop("pad_token_id", None),
            bos_token_id=kwargs.pop("bos_token_id", None),
            eos_token_id=kwargs.pop("eos_token_id", None),
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Qwen3Config"]
