# coding=utf-8
"""Standalone standard-PyTorch Gemma4 31B text model."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_gemma4_31b import Gemma4TextConfig, Gemma4Config


def _gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv, seq_len, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, num_kv, n_rep, seq_len, head_dim)
        .reshape(batch, num_kv * n_rep, seq_len, head_dim)
    )


class Gemma4RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.variance_epsilon = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_buffer("weight", torch.ones(hidden_size), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (hidden_states * self.weight).to(input_dtype)


class Gemma4RotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        sliding_params = config.rope_parameters["sliding_attention"]
        full_params = config.rope_parameters["full_attention"]
        sliding_dim = config.head_dim
        full_dim = config.global_head_dim
        partial_rotary_factor = full_params.get("partial_rotary_factor", 0.25)
        self.full_rotary_dim = int(full_dim * partial_rotary_factor)

        inv_freq_sliding = 1.0 / (
            sliding_params.get("rope_theta", 10000.0)
            ** (torch.arange(0, sliding_dim, 2, dtype=torch.float32) / sliding_dim)
        )
        full_factor = full_params.get("factor", 1.0)
        inv_freq_full = 1.0 / (
            full_params.get("rope_theta", 1000000.0)
            ** (torch.arange(0, self.full_rotary_dim, 2, dtype=torch.float32) / full_dim)
        ) / full_factor
        self.register_buffer("inv_freq_sliding", inv_freq_sliding, persistent=False)
        self.register_buffer("inv_freq_full", inv_freq_full, persistent=False)

    def forward(self, position_ids: torch.Tensor, layer_type: str, dtype: torch.dtype):
        inv_freq = self.inv_freq_sliding if layer_type == "sliding_attention" else self.inv_freq_full
        freqs = position_ids[..., None].float() * inv_freq.to(position_ids.device)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype).unsqueeze(1), emb.sin().to(dtype).unsqueeze(1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = (q_rot * cos) + (_rotate_half(q_rot.float()).to(q.dtype) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot.float()).to(k.dtype) * sin)
    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


class Gemma4MLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(_gelu_pytorch_tanh(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Gemma4Router(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        if config.num_experts is None or config.top_k_experts is None:
            raise ValueError("Gemma4 MoE requires num_experts and top_k_experts")
        self.hidden_size = config.hidden_size
        self.scalar_root_size = self.hidden_size ** -0.5
        self.top_k_experts = config.top_k_experts
        self.num_experts = config.num_experts
        self.norm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps, with_scale=False)
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size
        expert_scores = self.proj(hidden_states)
        topk_weight, topk_idx = torch.topk(expert_scores.softmax(dim=-1, dtype=torch.float32), k=self.top_k_experts, dim=-1)
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        topk_weight = topk_weight * self.per_expert_scale[topk_idx]
        return topk_idx, topk_weight.to(hidden_states.dtype)


class Gemma4SparseMoeBlock(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.router = Gemma4Router(config)
        self.experts = nn.ModuleList(
            [Gemma4MLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, original_shape[-1])
        topk_idx, topk_weight = self.router(flat_states)
        output = torch.zeros_like(flat_states)
        for expert_id, expert in enumerate(self.experts):
            mask = topk_idx == expert_id
            if not mask.any():
                continue
            token_idx = mask.nonzero(as_tuple=True)[0]
            expert_out = expert(flat_states[token_idx])
            weight = topk_weight[mask].unsqueeze(-1)
            output.index_add_(0, token_idx, (expert_out * weight).to(output.dtype))
        return output.reshape(original_shape)


class Gemma4Attention(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        if self.layer_type == "full_attention":
            self.num_kv_heads = config.num_global_key_value_heads
            self.head_dim = config.global_head_dim
        else:
            self.num_kv_heads = config.num_key_value_heads
            self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        sliding_attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        rotary_emb: Gemma4RotaryEmbedding,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        query = self.q_norm(query)
        key = self.k_norm(key)
        cos, sin = rotary_emb(position_ids, self.layer_type, query.dtype)
        query, key = _apply_rope(query, key, cos, sin)
        key = _repeat_kv(key, self.num_kv_groups)
        value = _repeat_kv(value, self.num_kv_groups)
        scores = torch.matmul(query, key.transpose(2, 3))
        mask = sliding_attention_mask if self.layer_type == "sliding_attention" else attention_mask
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
        weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
        output = torch.matmul(weights, value)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(output)


class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Gemma4Attention(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Gemma4SparseMoeBlock(config) if config.enable_moe_block else Gemma4MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        sliding_attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        rotary_emb: Gemma4RotaryEmbedding,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, sliding_attention_mask, position_ids, rotary_emb)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class Gemma4Model(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.embed_scale = config.hidden_size ** 0.5
        self.layers = nn.ModuleList([Gemma4DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Gemma4RotaryEmbedding(config)

    def _make_causal_mask(self, input_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        seq_len = input_ids.size(1)
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=input_ids.device, dtype=dtype)
        return torch.triu(mask, diagonal=1).view(1, 1, seq_len, seq_len)

    def _make_sliding_mask(self, input_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        seq_len = input_ids.size(1)
        causal = self._make_causal_mask(input_ids, dtype)
        positions = torch.arange(seq_len, device=input_ids.device)
        distance = positions.view(-1, 1) - positions.view(1, -1)
        window_mask = distance >= self.config.sliding_window
        return causal + window_mask.view(1, 1, seq_len, seq_len).to(dtype) * torch.finfo(dtype).min

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale
        position_ids = torch.arange(seq_len, device=input_ids.device).view(1, -1).expand(batch, -1)
        attention_mask = self._make_causal_mask(input_ids, hidden_states.dtype)
        sliding_attention_mask = self._make_sliding_mask(input_ids, hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, sliding_attention_mask, position_ids, self.rotary_emb)
        return self.norm(hidden_states)


class Gemma4ForCausalLM(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.model = Gemma4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            scale = self.config.final_logit_softcapping
            logits = torch.tanh(logits / scale) * scale
        return logits


__all__ = ["Gemma4Config", "Gemma4TextConfig", "Gemma4ForCausalLM", "Gemma4Model"]
