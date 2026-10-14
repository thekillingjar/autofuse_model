# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
"""Qwen2 configuration with the Hugging Face Qwen2.5-14B defaults."""

from typing import ClassVar

from transformers.configuration_utils import PretrainedConfig


class Qwen25_14BConfig(PretrainedConfig):
    model_type = "qwen2"
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["past_key_values"]

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=48,
        num_attention_heads=40,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=131072,
        max_window_layers=48,
        attention_dropout=0.0,
        **kwargs,
    ):
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        for name, value in (
            ("hidden_size", hidden_size),
            ("num_attention_heads", num_attention_heads),
            ("num_key_value_heads", num_key_value_heads),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if hidden_size % num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if num_attention_heads % num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if (hidden_size // num_attention_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
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
        # Transformers 5 converts the legacy rope_scaling field to rope_parameters.
        rope_parameters = kwargs.pop("rope_parameters", None)
        rope_parameters = dict(rope_parameters or rope_scaling or {})
        rope_parameters.setdefault("rope_type", rope_parameters.pop("type", "default"))
        rope_parameters.setdefault("rope_theta", rope_theta)
        self.rope_theta = rope_parameters["rope_theta"]
        self.rope_scaling = rope_scaling
        self.rope_parameters = rope_parameters
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.attention_dropout = attention_dropout
        super().__init__(
            pad_token_id=kwargs.pop("pad_token_id", None),
            bos_token_id=kwargs.pop("bos_token_id", 151643),
            eos_token_id=kwargs.pop("eos_token_id", 151643),
            tie_word_embeddings=tie_word_embeddings,
            rope_parameters=rope_parameters,
            **kwargs,
        )


Qwen2Config = Qwen25_14BConfig

__all__ = ["Qwen2Config", "Qwen25_14BConfig"]
