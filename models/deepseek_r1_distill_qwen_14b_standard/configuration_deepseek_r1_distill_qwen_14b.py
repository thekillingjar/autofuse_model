# coding=utf-8
"""Standalone configuration for DeepSeek-R1-Distill-Qwen-14B."""

from transformers.configuration_utils import PretrainedConfig


class DeepseekR1DistillQwen14BConfig(PretrainedConfig):
    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

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
        attention_bias=True,
        attention_dropout=0.0,
        sliding_window=131072,
        use_sliding_window=False,
        max_window_layers=48,
        **kwargs,
    ):
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
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.use_sliding_window = use_sliding_window
        self.max_window_layers = max_window_layers
        self.head_dim = hidden_size // num_attention_heads
        super().__init__(
            pad_token_id=kwargs.pop("pad_token_id", None),
            bos_token_id=kwargs.pop("bos_token_id", 151643),
            eos_token_id=kwargs.pop("eos_token_id", 151643),
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


Qwen2Config = DeepseekR1DistillQwen14BConfig

__all__ = ["DeepseekR1DistillQwen14BConfig", "Qwen2Config"]
