# coding=utf-8
"""Standard PyTorch BERT and ViT models for AutoFuse comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BertConfig:
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    num_labels: int = 2


@dataclass
class ViTConfig:
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-6
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    num_labels: int = 1000


class TransformerSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, attention_dropout: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_dropout)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, hidden = hidden_states.shape
        query = self._shape(self.q_proj(hidden_states))
        key = self._shape(self.k_proj(hidden_states))
        value = self._shape(self.v_proj(hidden_states))
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = F.softmax(scores.float(), dim=-1).to(query.dtype)
        probs = self.dropout(probs)
        context = torch.matmul(probs, value)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        return self.out_proj(context)


class TransformerMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states), approximate="tanh"))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, eps: float, dropout: float, attn_dropout: float):
        super().__init__()
        self.attention = TransformerSelfAttention(hidden_size, num_heads, attn_dropout)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = TransformerMLP(hidden_size, intermediate_size)
        self.mlp_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.attention_norm(hidden_states + self.dropout(attention_output))
        mlp_output = self.mlp(hidden_states)
        return self.mlp_norm(hidden_states + self.dropout(mlp_output))


class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).view(1, -1).expand(batch, -1)
        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.dropout(self.layer_norm(embeddings))


class BertForSequenceClassification(nn.Module):
    def __init__(self, config: BertConfig | None = None):
        super().__init__()
        self.config = config or BertConfig()
        self.embeddings = BertEmbeddings(self.config)
        self.encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    self.config.hidden_size,
                    self.config.num_attention_heads,
                    self.config.intermediate_size,
                    self.config.layer_norm_eps,
                    self.config.hidden_dropout_prob,
                    self.config.attention_probs_dropout_prob,
                )
                for _ in range(self.config.num_hidden_layers)
            ]
        )
        self.pooler = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids)
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask.to(hidden_states.dtype))[:, None, None, :]
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
        for layer in self.encoder:
            hidden_states = layer(hidden_states, attention_mask)
        pooled = torch.tanh(self.pooler(hidden_states[:, 0]))
        return self.classifier(pooled)


class ViTPatchEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.projection(pixel_values)
        return embeddings.flatten(2).transpose(1, 2)


class ViTForImageClassification(nn.Module):
    def __init__(self, config: ViTConfig | None = None):
        super().__init__()
        self.config = config or ViTConfig()
        image_size = self.config.image_size
        patch_size = self.config.patch_size
        num_patches = (image_size // patch_size) ** 2
        self.patch_embeddings = ViTPatchEmbeddings(self.config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, self.config.hidden_size))
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    self.config.hidden_size,
                    self.config.num_attention_heads,
                    self.config.intermediate_size,
                    self.config.layer_norm_eps,
                    self.config.hidden_dropout_prob,
                    self.config.attention_probs_dropout_prob,
                )
                for _ in range(self.config.num_hidden_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch = pixel_values.shape[0]
        hidden_states = self.patch_embeddings(pixel_values)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)
        hidden_states = self.dropout(hidden_states + self.position_embeddings)
        for layer in self.encoder:
            hidden_states = layer(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return self.classifier(hidden_states[:, 0])


MODEL_BUILDERS: dict[str, Callable[[], nn.Module]] = {
    "bert": BertForSequenceClassification,
    "vit": ViTForImageClassification,
}


def build_model(model_type: str, num_layers: int | None = None, seq_len_or_image_size: int | None = None) -> nn.Module:
    if model_type == "bert":
        config = BertConfig()
        if num_layers is not None:
            config.num_hidden_layers = num_layers
        if seq_len_or_image_size is not None:
            config.max_position_embeddings = max(config.max_position_embeddings, seq_len_or_image_size)
        return BertForSequenceClassification(config)
    if model_type == "vit":
        config = ViTConfig()
        if num_layers is not None:
            config.num_hidden_layers = num_layers
        if seq_len_or_image_size is not None:
            config.image_size = seq_len_or_image_size
        return ViTForImageClassification(config)
    raise ValueError(f"Unsupported model_type: {model_type}. Choices: {sorted(MODEL_BUILDERS)}")


__all__ = ["build_model", "BertForSequenceClassification", "ViTForImageClassification"]
