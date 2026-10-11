# coding=utf-8
"""Standalone standard-PyTorch DeepSeek-MoE 16B model."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_deepseek_moe_16b import DeepseekConfig, DeepseekMoeConfig


class DeepseekRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class DeepseekRotaryEmbedding(nn.Module):
    def __init__(self, config: DeepseekConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        freqs = positions[..., None].float() * self.inv_freq.to(q.device)
        cos = freqs.cos().to(q.dtype)
        sin = freqs.sin().to(q.dtype)
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

        q_embed = (q * cos) + (_rotate_half(q.float()).to(q.dtype) * sin)
        k_embed = (k * cos) + (_rotate_half(k.float()).to(k.dtype) * sin)
        return q_embed, k_embed


class DeepseekMLP(nn.Module):
    def __init__(
        self,
        config: DeepseekConfig,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        hidden_size = hidden_size or config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=config.mlp_bias)
        self.act_fn = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DeepseekMoEGate(nn.Module):
    def __init__(self, config: DeepseekConfig):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.n_routed_experts, self.hidden_size))
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.view(-1, self.hidden_size).float()
        return F.linear(x, self.weight)


class DeepseekMoE(nn.Module):
    def __init__(self, config: DeepseekConfig):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        self.experts = nn.ModuleList(
            [
                DeepseekMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(self.n_routed_experts)
            ]
        )
        self.gate = DeepseekMoEGate(config)
        shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
        self.shared_experts = DeepseekMLP(config, intermediate_size=shared_intermediate)

    def forward(self, hidden_states: torch.Tensor):
        orig_shape = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, self.config.hidden_size)

        logits = self.gate(flat_hidden)
        weights = logits.softmax(dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(
            weights, k=self.num_experts_per_tok, dim=-1, sorted=False
        )
        if self.norm_topk_prob:
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        topk_weight = topk_weight * self.routed_scaling_factor
        topk_weight = topk_weight.to(flat_hidden.dtype)

        routed = torch.zeros_like(flat_hidden)
        for i, expert in enumerate(self.experts):
            mask = topk_idx == i
            if not mask.any():
                continue
            flat_idx = mask.nonzero(as_tuple=True)[0]
            expert_out = expert(flat_hidden[flat_idx])
            w = topk_weight[mask].unsqueeze(-1)
            routed.index_add_(0, flat_idx, (expert_out * w).to(routed.dtype))

        out = routed.reshape(orig_shape) + self.shared_experts(hidden_states)
        return out, logits


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv, slen, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, num_kv, n_rep, slen, head_dim)
        .reshape(batch, num_kv * n_rep, slen, head_dim)
    )


class DeepseekAttention(nn.Module):
    def __init__(self, config: DeepseekConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** (-0.5)
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        rotary_emb: DeepseekRotaryEmbedding,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        q, k = rotary_emb(q, k, position_ids)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        k = _repeat_kv(k, self.num_kv_groups)
        v = _repeat_kv(v, self.num_kv_groups)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)


class DeepseekDecoderLayer(nn.Module):
    def __init__(self, config: DeepseekConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.input_layernorm = DeepseekRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = DeepseekAttention(config, layer_idx)
        self.post_attention_layernorm = DeepseekRMSNorm(config.hidden_size, config.rms_norm_eps)

        if layer_idx < config.first_k_dense_replace:
            self.mlp = DeepseekMLP(config)
        else:
            self.mlp = DeepseekMoE(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        rotary_emb: DeepseekRotaryEmbedding,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask, position_ids, rotary_emb)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = residual + hidden_states
        return hidden_states


class DeepseekModel(nn.Module):
    def __init__(self, config: DeepseekConfig):
        super().__init__()
        self.config = config
        self.padding_idx = 0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.layers = nn.ModuleList(
            [DeepseekDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = DeepseekRotaryEmbedding(config)

    def _make_causal_mask(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        seq_len = x.size(1)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None, :, :].to(dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).expand(bsz, -1)
        attention_mask = self._make_causal_mask(input_ids, hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids, self.rotary_emb)
        return self.norm(hidden_states)


class DeepseekForCausalLM(nn.Module):
    def __init__(self, config: DeepseekConfig):
        super().__init__()
        self._config = config
        self.model = DeepseekModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @property
    def config(self) -> DeepseekConfig:
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids)
        return self.lm_head(hidden_states)


DeepseekMoeForCausalLM = DeepseekForCausalLM
DeepseekMoeModel = DeepseekModel

__all__ = [
    "DeepseekConfig",
    "DeepseekMoeConfig",
    "DeepseekForCausalLM",
    "DeepseekMoeForCausalLM",
    "DeepseekModel",
    "DeepseekMoeModel",
]
