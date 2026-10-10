# coding=utf-8
"""Standalone standard-PyTorch DeepSeek-MoE 16B.

This reference intentionally avoids the repository executor, module package,
torch_npu fused operators, and custom MoE kernels. It is intended for comparing
plain eager execution with torch.compile + AutoFuse.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_deepseek_moe_16b import DeepseekMoeConfig


def rms_norm(x, weight, eps):
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


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


class DeepseekMoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)


class DeepseekMoeRotaryEmbedding(nn.Module):
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


class DeepseekMoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size)
        )

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.float(), self.weight.float())
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise ValueError(f"Unsupported scoring function: {self.scoring_func}")
        topk_weight, topk_idx = torch.topk(
            scores,
            k=self.top_k,
            dim=-1,
            sorted=False,
        )
        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight.to(hidden_states.dtype)


class DeepseekMoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = MoEGate(config)
        self.experts = nn.ModuleList(
            [
                DeepseekMoeMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.shared_experts = None
        if config.n_shared_experts is not None and config.n_shared_experts > 0:
            self.shared_experts = DeepseekMoeMLP(
                config,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
            )

    def _grouped_experts(self, hidden_states, tokens_per_expert):
        """Run all routed experts through NPU grouped matmul kernels."""
        try:
            import torch_npu
        except ImportError as exc:
            raise RuntimeError("Dynamic MoE execution requires torch_npu") from exc

        gate_up_weight = torch.stack(
            [torch.cat((expert.gate_proj.weight, expert.up_proj.weight), dim=0)
             for expert in self.experts],
            dim=0,
        )
        down_weight = torch.stack(
            [expert.down_proj.weight for expert in self.experts],
            dim=0,
        )
        gate_up = torch_npu.npu_grouped_matmul(
            [hidden_states],
            [gate_up_weight],
            group_list=tokens_per_expert,
            split_item=3,
            group_type=0,
            group_list_type=1,
            output_dtype=hidden_states.dtype,
            tuning_config=[0],
        )[0]
        intermediate_size = self.experts[0].gate_proj.out_features
        gate, up = gate_up.split(intermediate_size, dim=-1)
        activated = F.silu(gate) * up
        return torch_npu.npu_grouped_matmul(
            [activated],
            [down_weight],
            group_list=tokens_per_expert,
            split_item=3,
            group_type=0,
            group_list_type=1,
            output_dtype=hidden_states.dtype,
            tuning_config=[0],
        )[0]

    def forward(self, hidden_states):
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, original_shape[-1])
        topk_idx, topk_weight = self.gate(flat_states)
        import torch_npu

        expanded_states, expanded_row_idx, tokens_per_expert, _ = (
            torch_npu.npu_moe_init_routing_v2(
                flat_states,
                expert_idx=topk_idx.to(torch.int32),
                active_num=topk_idx.shape[0] * topk_idx.shape[1],
                expert_num=self.gate.n_routed_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, self.gate.n_routed_experts],
                quant_mode=-1,
            )
        )
        expert_output = self._grouped_experts(expanded_states, tokens_per_expert)
        output = torch_npu.npu_moe_finalize_routing(
            expert_output,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weight.to(expert_output.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None,
            drop_pad_mode=2,
        )
        if self.shared_experts is not None:
            output = output + self.shared_experts(flat_states)
        return output.reshape(original_shape)


class DeepseekMoeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
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
            bias=config.attention_bias,
        )

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
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if self.num_key_value_groups > 1:
            key = key.repeat_interleave(self.num_key_value_groups, dim=1)
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        scores = scores + attention_mask
        weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
        output = torch.matmul(weights, value)
        output = output.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(output)


class DeepseekMoeDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = DeepseekMoeAttention(config)
        self.mlp = (
            DeepseekMoeMLP(config)
            if layer_idx < config.first_k_dense_replace
            else DeepseekMoeSparseMoeBlock(config)
        )
        self.input_layernorm = DeepseekMoeRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.post_attention_layernorm = DeepseekMoeRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

    def forward(self, hidden_states, cos, sin, attention_mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class DeepseekMoePreTrainedModel(PreTrainedModel):
    config_class = DeepseekMoeConfig
    base_model_prefix = "model"
    _no_split_modules = ["DeepseekMoeDecoderLayer"]


class DeepseekMoeModel(DeepseekMoePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [DeepseekMoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekMoeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = DeepseekMoeRotaryEmbedding(config)
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


class DeepseekMoeForCausalLM(DeepseekMoePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekMoeModel(config)
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


__all__ = [
    "DeepseekMoeConfig",
    "DeepseekMoeForCausalLM",
    "DeepseekMoeModel",
]
