# coding=utf-8
"""Configuration for the standalone DeepSeek-MoE 16B reference model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DeepseekConfig:
    """DeepSeek-MoE config, defaulting to deepseek-moe-16b-base shapes."""

    vocab_size: int = 102400
    hidden_size: int = 2048
    intermediate_size: int = 1096
    moe_intermediate_size: int = 2816
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    n_routed_experts: int = 64
    num_experts_per_tok: int = 6
    n_shared_experts: int = 2
    first_k_dense_replace: int = 1
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    norm_topk_prob: bool = False
    routed_scaling_factor: float = 1.0
    rope_scaling: Optional[dict] = None
    attention_dropout: float = 0.0

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} must be divisible by "
                f"num_attention_heads {self.num_attention_heads}"
            )
        if self.n_routed_experts <= 0:
            raise ValueError("n_routed_experts must be positive")

    @classmethod
    def from_dict(cls, kwargs):
        params = dict(kwargs)
        valid_keys = set(cls.__dataclass_fields__)
        params = {key: value for key, value in params.items() if key in valid_keys}
        return cls(**params)

    @classmethod
    def from_pretrained(cls, model_path):
        config_path = Path(model_path) / "config.json"
        with config_path.open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


DeepseekMoeConfig = DeepseekConfig

__all__ = ["DeepseekConfig", "DeepseekMoeConfig"]
