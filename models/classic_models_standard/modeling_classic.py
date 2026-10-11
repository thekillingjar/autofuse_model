# coding=utf-8
"""Standard PyTorch classic models for AutoFuse comparison."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dims: list[int], activation: Callable[[torch.Tensor], torch.Tensor] = F.relu):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class AFN(nn.Module):
    """Adaptive Factorization Network style CTR model."""

    def __init__(self, num_fields: int = 20, vocab_size: int = 10000, embed_dim: int = 16, hidden_dim: int = 256):
        super().__init__()
        self.num_fields = num_fields
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.log_transform = nn.Linear(num_fields, hidden_dim)
        self.deep = MLP([num_fields * embed_dim, hidden_dim, hidden_dim, 1])
        self.out = nn.Linear(hidden_dim + 1, 1)

    def forward(self, feature_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embed(feature_ids)
        log_emb = torch.log(torch.clamp(torch.abs(emb), min=1e-6))
        lnn = self.log_transform(log_emb.transpose(1, 2)).transpose(1, 2)
        lnn = torch.exp(lnn).mean(dim=1)
        deep = self.deep(emb.flatten(1))
        return self.out(torch.cat([lnn, deep], dim=-1))


class ESMM(nn.Module):
    """Entire Space Multi-task Model for CTR/CVR."""

    def __init__(self, num_fields: int = 20, vocab_size: int = 10000, embed_dim: int = 16, hidden_dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        input_dim = num_fields * embed_dim
        self.ctr_tower = MLP([input_dim, hidden_dim, hidden_dim, 1])
        self.cvr_tower = MLP([input_dim, hidden_dim, hidden_dim, 1])

    def forward(self, feature_ids: torch.Tensor):
        x = self.embed(feature_ids).flatten(1)
        ctr = torch.sigmoid(self.ctr_tower(x))
        cvr = torch.sigmoid(self.cvr_tower(x))
        ctcvr = ctr * cvr
        return torch.cat([ctr, cvr, ctcvr], dim=-1)


class AttentionSequencePooling(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = MLP([embed_dim * 4, hidden_dim, hidden_dim, 1])

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        query = query[:, None, :].expand_as(keys)
        attn_input = torch.cat([query, keys, query - keys, query * keys], dim=-1)
        scores = self.mlp(attn_input).squeeze(-1)
        scores = scores + (1.0 - mask.to(scores.dtype)) * torch.finfo(scores.dtype).min
        weights = F.softmax(scores.float(), dim=-1).to(keys.dtype)
        return torch.matmul(weights[:, None, :], keys).squeeze(1)


class DIEN(nn.Module):
    """Deep Interest Evolution Network style model."""

    def __init__(self, vocab_size: int = 10000, embed_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.item_embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.query_proj = nn.Linear(embed_dim, hidden_dim)
        self.attn = AttentionSequencePooling(hidden_dim, hidden_dim)
        self.mlp = MLP([hidden_dim + embed_dim, hidden_dim, hidden_dim, 1])

    def forward(self, history_ids: torch.Tensor, target_id: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        history = self.item_embed(history_ids)
        target = self.item_embed(target_id)
        seq_out, _ = self.gru(history)
        interest = self.attn(self.query_proj(target), seq_out, mask)
        return self.mlp(torch.cat([interest, target], dim=-1))


class SimpleTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP([hidden_size, hidden_size * mlp_ratio, hidden_size])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=attn_mask, need_weights=False)
        x = x + y
        return x + self.mlp(self.norm2(x))


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim: int = 17, act_dim: int = 6, hidden_size: int = 128, num_layers: int = 4, num_heads: int = 4, max_len: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_embed = nn.Linear(state_dim, hidden_size)
        self.action_embed = nn.Linear(act_dim, hidden_size)
        self.return_embed = nn.Linear(1, hidden_size)
        self.time_embed = nn.Embedding(max_len, hidden_size)
        self.blocks = nn.ModuleList([SimpleTransformerBlock(hidden_size, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.action_head = nn.Linear(hidden_size, act_dim)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, returns_to_go: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embed(timesteps)
        x = torch.stack(
            [
                self.return_embed(returns_to_go) + time_emb,
                self.state_embed(states) + time_emb,
                self.action_embed(actions) + time_emb,
            ],
            dim=2,
        ).flatten(1, 2)
        seq_len = x.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            x = block(x, causal_mask)
        x = self.norm(x)
        state_tokens = x[:, 1::3]
        return self.action_head(state_tokens)


class SAM(nn.Module):
    """Small Segment-Anything-style model for static AutoFuse experiments."""

    def __init__(self, image_size: int = 256, patch_size: int = 16, hidden_size: int = 256, num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        self.blocks = nn.ModuleList([SimpleTransformerBlock(hidden_size, num_heads) for _ in range(num_layers)])
        self.prompt_embed = nn.Linear(2, hidden_size)
        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, hidden_size // 2, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_size // 2, hidden_size // 4, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(hidden_size // 4, 1, kernel_size=1),
        )

    def forward(self, images: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2) + self.pos_embed
        prompt = self.prompt_embed(points).mean(dim=1, keepdim=True)
        x = x + prompt
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2).reshape(images.shape[0], -1, height, width)
        return self.mask_head(x)


MODEL_BUILDERS: dict[str, Callable[[], nn.Module]] = {
    "afn": AFN,
    "esmm": ESMM,
    "dien": DIEN,
    "decision_transformer": DecisionTransformer,
    "sam": SAM,
}


def build_model(model_type: str) -> nn.Module:
    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported model_type: {model_type}. Choices: {sorted(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[model_type]()


__all__ = ["build_model", "MODEL_BUILDERS", "AFN", "ESMM", "DIEN", "DecisionTransformer", "SAM"]
