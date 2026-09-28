# coding=utf-8
"""Standalone, standard-PyTorch Qwen3.5-9B text model.

No repository module, Ascend operator, fused attention, parallel linear layer,
or custom KV-cache implementation is used in this file. The implementation is
deliberately ordinary PyTorch so the only experimental variable is the
torch.compile backend used by the benchmark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_qwen3_5_9b import Qwen3_5TextConfig


def rms_norm(x, weight, eps):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * (1.0 + weight.float())).to(dtype=x.dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q, k, cos, sin):
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    return (torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1))


def causal_mask(batch, query_len, key_len, device, dtype):
    past_len = key_len - query_len
    mask = torch.full(
        (query_len, key_len),
        torch.finfo(dtype).min,
        device=device,
        dtype=dtype,
    )
    mask = torch.triu(mask, diagonal=1 + past_len)
    return mask.view(1, 1, query_len, key_len).expand(batch, 1, query_len, key_len)


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


class Qwen3_5RMSNormGated(Qwen3_5RMSNorm):
    def forward(self, x, gate):
        return rms_norm(x, self.weight, self.eps) * F.silu(gate)


class Qwen3_5RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        rope = config.rope_parameters
        dim = int(config.head_dim * rope.get("partial_rotary_factor", 0.25))
        inv_freq = 1.0 / (
            rope.get("rope_theta", 1000000.0)
            ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, dtype):
        if position_ids.ndim == 3:
            position_ids = position_ids[0]
        freqs = torch.einsum("bs,d->bsd", position_ids.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype).unsqueeze(1), emb.sin().to(dtype=dtype).unsqueeze(1)


class Qwen3_5Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, x, cos, sin, attention_mask=None):
        batch, seq_len, _ = x.shape
        q, gate = self.q_proj(x).view(
            batch, seq_len, self.num_heads, 2 * self.head_dim
        ).chunk(2, dim=-1)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attention_mask is None:
            scores = scores + causal_mask(batch, seq_len, seq_len, x.device, scores.dtype)
        else:
            scores = scores + attention_mask
        probs = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        output = torch.matmul(probs, v).transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(output * torch.sigmoid(gate.reshape(batch, seq_len, -1)))


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        self.in_proj_qkvz = nn.Linear(
            self.hidden_size,
            self.conv_dim + self.value_dim,
            bias=False,
        )
        self.in_proj_ba = nn.Linear(
            self.hidden_size,
            self.num_v_heads * 2,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        projected = self.in_proj_qkvz(x)
        mixed, z = projected.split((self.conv_dim, self.value_dim), dim=-1)
        b, a = self.in_proj_ba(x).chunk(2, dim=-1)
        mixed = mixed.transpose(1, 2)
        mixed = self.conv1d(mixed)[..., :seq_len]
        mixed = F.silu(mixed).transpose(1, 2)
        q, k, v = mixed.split((self.key_dim, self.key_dim, self.value_dim), dim=-1)
        q = q.view(batch, seq_len, self.num_k_heads, self.head_k_dim)
        k = k.view(batch, seq_len, self.num_k_heads, self.head_k_dim)
        v = v.view(batch, seq_len, self.num_v_heads, self.head_v_dim)
        q = q.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        k = k.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        q = F.normalize(q.float(), dim=-1).to(dtype=x.dtype)
        k = F.normalize(k.float(), dim=-1).to(dtype=x.dtype)
        beta = b.sigmoid().float()
        decay = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        state = torch.zeros(
            batch,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            device=x.device,
            dtype=torch.float32,
        )
        outputs = []
        q = q.float() / math.sqrt(self.head_k_dim)
        k = k.float()
        v = v.float()
        for index in range(seq_len):
            state = state * decay[:, index, :, None, None].exp()
            q_t = q[:, index]
            k_t = k[:, index]
            v_t = v[:, index]
            predicted = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - predicted) * beta[:, index, :, None]
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            outputs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))
        output = torch.stack(outputs, dim=1).to(dtype=x.dtype)
        output = output.reshape(batch, seq_len, self.value_dim)
        z = z.view(batch, seq_len, self.num_v_heads, self.head_v_dim)
        output = self.norm(output.view(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(output.view(batch, seq_len, self.value_dim))


class Qwen3_5MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config)
        else:
            self.self_attn = Qwen3_5Attention(config)
        self.mlp = Qwen3_5MLP(config)

    def forward(self, x, cos, sin, attention_mask=None):
        residual = x
        x = self.input_layernorm(x)
        if self.layer_type == "linear_attention":
            x = self.linear_attn(x)
        else:
            x = self.self_attn(x, cos, sin, attention_mask)
        x = residual + x
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3_5PreTrainedModel(PreTrainedModel):
    config_class = Qwen3_5TextConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen3_5DecoderLayer"]


class Qwen3_5TextModel(Qwen3_5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, index) for index in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3_5RotaryEmbedding(config)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        x = self.embed_tokens(input_ids)
        batch, seq_len, _ = x.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).view(1, -1).expand(batch, -1)
        if attention_mask is not None and attention_mask.ndim == 2:
            padding = (1.0 - attention_mask.to(dtype=x.dtype))[:, None, None, :]
            attention_mask = padding * torch.finfo(x.dtype).min
            attention_mask = attention_mask + causal_mask(
                batch,
                seq_len,
                seq_len,
                x.device,
                x.dtype,
            )
        cos, sin = self.rotary_emb(position_ids, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask)
        return BaseModelOutputWithPast(last_hidden_state=self.norm(x), past_key_values=None)


class Qwen3_5ForCausalLM(Qwen3_5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
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


__all__ = ["Qwen3_5ForCausalLM", "Qwen3_5TextModel", "Qwen3_5TextConfig"]
