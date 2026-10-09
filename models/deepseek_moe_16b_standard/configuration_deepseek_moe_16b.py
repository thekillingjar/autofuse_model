# coding=utf-8
"""Configuration for the standalone DeepSeek-MoE 16B reference model."""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig


class DeepseekMoeConfig(PretrainedConfig):
    model_type = "deepseek"

    def __init__(
        self,
        vocab_size=102400,
        hidden_size=2048,
        intermediate_size=10944,
        moe_intermediate_size=1408,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=100000,
        eos_token_id=100001,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_bias=False,
        scoring_func="softmax",
        num_experts_per_tok=6,
        n_routed_experts=64,
        n_shared_experts=2,
        routed_scaling_factor=1.0,
        first_k_dense_replace=1,
        norm_topk_prob=False,
        aux_loss_alpha=0.001,
        seq_aux=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.scoring_func = scoring_func
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.head_dim = hidden_size // num_attention_heads
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["DeepseekMoeConfig"]
