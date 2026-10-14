# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
"""Qwen2.5-14B no-cache reference based on Transformers' eager Qwen2.

Only standard PyTorch operations are used. No repository model, NPU fused
operator, SDPA, FlashAttention, executor, or parallel layer is imported.
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel

from .configuration_qwen2_5_14b import Qwen25_14BConfig

try:
    from transformers import initialization as init
except ImportError:
    from torch.nn import init


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.rope_parameters["rope_type"] != "default":
            raise ValueError(
                "This benchmark supports the checkpoint's unscaled RoPE only"
            )
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = config.rope_theta
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply(self, fn, recurse=True):
        # Keep frequencies in FP32 even when random-init parameters are cast to BF16.
        inv_freq = self.inv_freq
        result = super()._apply(fn, recurse=recurse)
        self.inv_freq = inv_freq.to(device=self.inv_freq.device, dtype=torch.float32)
        return result

    def forward(self, hidden_states, position_ids):
        inv_freq = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        positions = position_ids[:, None, :].float()
        freqs = torch.matmul(inv_freq, positions).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(hidden_states.dtype), emb.sin().to(hidden_states.dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(query, key, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return query * cos + rotate_half(query) * sin, key * cos + rotate_half(key) * sin


def repeat_kv(hidden_states, repetitions):
    if repetitions == 1:
        return hidden_states
    batch, heads, seq_len, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, heads, repetitions, seq_len, head_dim)
        .reshape(batch, heads * repetitions, seq_len, head_dim)
    )


def make_causal_mask(hidden_states, attention_mask):
    batch, seq_len, _ = hidden_states.shape
    if attention_mask is not None and attention_mask.ndim == 4:
        return attention_mask.to(dtype=hidden_states.dtype, device=hidden_states.device)
    min_dtype = torch.finfo(hidden_states.dtype).min
    mask = torch.full(
        (seq_len, seq_len),
        min_dtype,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    ).triu(diagonal=1)
    mask = mask[None, None, :, :].expand(batch, 1, seq_len, seq_len)
    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape != (batch, seq_len):
            raise ValueError(
                "attention_mask must be [batch, seq_len] or a 4D additive mask"
            )
        # Mask rather than add two min_dtype values, matching HF even on left padding.
        padding_mask = (mask + attention_mask[:, None, None, :].to(mask.dtype)) == 0
        mask = mask.masked_fill(padding_mask, min_dtype)
    return mask


class Qwen2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, cos, sin, attention_mask):
        batch, seq_len, _ = hidden_states.shape
        query = (
            self.q_proj(hidden_states)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(hidden_states)
            .view(batch, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(hidden_states)
            .view(batch, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        weights = F.softmax(scores + attention_mask, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
        output = torch.matmul(weights, value)
        output = output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
        return self.o_proj(output)


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(self, hidden_states, cos, sin, attention_mask):
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), cos, sin, attention_mask
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen25_14BConfig
    base_model_prefix = "model"
    _no_split_modules: ClassVar[list[str]] = ["Qwen2DecoderLayer"]

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                init.zeros_(module.bias)
            if (
                isinstance(module, nn.Embedding)
                and module.padding_idx is not None
                and not getattr(module.weight, "_is_hf_initialized", False)
            ):
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, Qwen2RMSNorm):
            init.ones_(module.weight)
        elif isinstance(module, Qwen2RotaryEmbedding):
            # Non-persistent RoPE buffers also need initialization after meta loading.
            inv_freq = 1.0 / (
                module.rope_theta
                ** (
                    torch.arange(
                        0,
                        module.head_dim,
                        2,
                        device=module.inv_freq.device,
                        dtype=torch.float32,
                    )
                    / module.head_dim
                )
            )
            module.inv_freq.copy_(inv_freq)


class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if config.use_sliding_window:
            raise ValueError(
                "This benchmark implements Qwen2.5-14B full attention, not sliding windows"
            )
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        use_cache=False,
        past_key_values=None,
        return_dict=True,
    ):
        if use_cache or past_key_values is not None:
            raise ValueError(
                "This fixed-shape prefill benchmark requires use_cache=False and no past_key_values"
            )
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")
        hidden_states = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        seq_len = hidden_states.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(
                0
            )
        mask = make_causal_mask(hidden_states, attention_mask)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, mask)
        hidden_states = self.norm(hidden_states)
        if not return_dict:
            return (hidden_states,)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys: ClassVar[list[str]] = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        use_cache=False,
        past_key_values=None,
        return_dict=True,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=False,
        )[0]
        logits = self.lm_head(hidden_states)
        if not return_dict:
            return (logits,)
        return CausalLMOutputWithPast(logits=logits)


Qwen25_14BForCausalLM = Qwen2ForCausalLM
Qwen25_14BModel = Qwen2Model

__all__ = [
    "Qwen2ForCausalLM",
    "Qwen2Model",
    "Qwen25_14BConfig",
    "Qwen25_14BForCausalLM",
    "Qwen25_14BModel",
]
