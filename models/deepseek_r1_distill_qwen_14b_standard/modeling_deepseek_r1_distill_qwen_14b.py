# coding=utf-8
"""Standalone standard-PyTorch DeepSeek-R1-Distill-Qwen-14B model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_deepseek_r1_distill_qwen_14b import DeepseekR1DistillQwen14BConfig


def rms_norm(x, weight, eps):
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(q, k, cos, sin):
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


def repeat_kv(hidden_states, num_key_value_groups):
    if num_key_value_groups == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(
        batch,
        num_kv_heads,
        num_key_value_groups,
        seq_len,
        head_dim,
    ).reshape(batch, num_kv_heads * num_key_value_groups, seq_len, head_dim)


def make_causal_mask(batch, seq_len, device, dtype):
    mask = torch.full(
        (seq_len, seq_len),
        torch.finfo(dtype).min,
        device=device,
        dtype=dtype,
    )
    return torch.triu(mask, diagonal=1).view(1, 1, seq_len, seq_len).expand(
        batch,
        1,
        seq_len,
        seq_len,
    )


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, dtype):
        freqs = torch.einsum("bs,d->bsd", position_ids.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype).unsqueeze(1), emb.sin().to(dtype).unsqueeze(1)


class Qwen2Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.attention_dropout = config.attention_dropout

    def forward(self, hidden_states, cos, sin, attention_mask):
        batch, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        query, key = apply_rotary(query, key, cos, sin)
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        scores = scores + attention_mask
        weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
        weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
        output = torch.matmul(weights, value)
        output = output.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(output)


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = Qwen2Attention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, cos, sin, attention_mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = DeepseekR1DistillQwen14BConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen2DecoderLayer"]


class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        hidden_states = self.embed_tokens(input_ids)
        batch, seq_len, _ = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(
                seq_len,
                device=hidden_states.device,
            ).view(1, -1).expand(batch, -1)
        if attention_mask is None:
            attention_mask = make_causal_mask(
                batch,
                seq_len,
                hidden_states.device,
                hidden_states.dtype,
            )
        elif attention_mask.ndim == 2:
            padding_mask = (
                1.0 - attention_mask.to(hidden_states.dtype)
            )[:, None, None, :]
            attention_mask = padding_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask + make_causal_mask(
                batch,
                seq_len,
                hidden_states.device,
                hidden_states.dtype,
            )
        cos, sin = self.rotary_emb(position_ids, hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, attention_mask)
        return BaseModelOutputWithPast(
            last_hidden_state=self.norm(hidden_states),
            past_key_values=None,
        )


class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return CausalLMOutputWithPast(
            logits=self.lm_head(outputs.last_hidden_state),
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


DeepseekR1DistillQwen14BForCausalLM = Qwen2ForCausalLM
DeepseekR1DistillQwen14BModel = Qwen2Model

__all__ = [
    "DeepseekR1DistillQwen14BConfig",
    "DeepseekR1DistillQwen14BForCausalLM",
    "DeepseekR1DistillQwen14BModel",
    "Qwen2ForCausalLM",
    "Qwen2Model",
]
