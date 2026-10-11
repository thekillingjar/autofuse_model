# coding=utf-8
"""Configuration for the standalone Gemma4 31B AutoFuse comparison model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Gemma4TextConfig:
    vocab_size: int = 262144
    hidden_size: int = 2816
    intermediate_size: int = 2112
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2
    tie_word_embeddings: bool = True
    rope_parameters: Optional[dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: int = 1024
    layer_types: Optional[list[str]] = None
    final_logit_softcapping: Optional[float] = None
    num_global_key_value_heads: Optional[int] = None
    global_head_dim: int = 512
    attention_k_eq_v: bool = False
    enable_moe_block: bool = False
    num_experts: Optional[int] = None
    top_k_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None

    def __post_init__(self):
        if self.num_global_key_value_heads is None:
            self.num_global_key_value_heads = self.num_key_value_heads
        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                },
            }
        if self.layer_types is None:
            sliding_window_pattern = 6
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.layer_types and self.layer_types[-1] != "full_attention":
            self.layer_types[-1] = "full_attention"
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

    @classmethod
    def from_dict(cls, config_dict):
        data = dict(config_dict)
        if "text_config" in data:
            data = dict(data["text_config"])
        valid_keys = set(cls.__dataclass_fields__)
        data = {key: value for key, value in data.items() if key in valid_keys}
        return cls(**data)

    @classmethod
    def from_pretrained(cls, model_path):
        config_path = Path(model_path) / "config.json"
        with config_path.open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @property
    def num_experts_per_tok(self):
        return self.top_k_experts


Gemma4Config = Gemma4TextConfig

__all__ = ["Gemma4TextConfig", "Gemma4Config"]
