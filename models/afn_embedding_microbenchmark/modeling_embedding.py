"""Reproduce AFN's per-field Slice -> Cast -> Embedding without DeepCTR."""

from __future__ import annotations

import torch
from torch import nn

CASES = (
    "slice_cast_embedding",
    "cast_embedding",
    "embedding_only",
    "slice_cast",
    "slice_only",
    "slice_copy",
)


class AFNEmbeddingLookup(nn.Module):
    def __init__(
        self,
        num_fields=1,
        total_fields=39,
        field_offset=13,
        vocab_size=10000,
        embed_dim=128,
        case="slice_cast_embedding",
    ):
        super().__init__()
        if case not in CASES:
            raise ValueError(f"Unknown case: {case}")
        if min(num_fields, total_fields, vocab_size, embed_dim) <= 0:
            raise ValueError(
                "Field counts, vocabulary size, and embedding dimension must be positive"
            )
        if field_offset < 0 or field_offset + num_fields > total_fields:
            raise ValueError("Selected sparse fields must fit inside total_fields")
        self.case = case
        self.feature_index = {
            f"C{i + 1}": (field_offset + i, field_offset + i + 1)
            for i in range(num_fields)
        }
        self.total_fields = total_fields
        self.embedding_dict = (
            nn.ModuleDict(
                {
                    name: nn.Embedding(vocab_size, embed_dim)
                    for name in self.feature_index
                }
            )
            if case in ("slice_cast_embedding", "cast_embedding", "embedding_only")
            else nn.ModuleDict()
        )

    def forward(self, x):
        outputs = []
        for i, (name, (start, end)) in enumerate(self.feature_index.items()):
            if self.case == "slice_cast_embedding":
                output = self.embedding_dict[name](x[:, start:end].long())
            elif self.case == "cast_embedding":
                output = self.embedding_dict[name](x[i].long())
            elif self.case == "embedding_only":
                output = self.embedding_dict[name](x[i])
            elif self.case == "slice_cast":
                output = x[:, start:end].long()
            elif self.case == "slice_only":
                output = x[:, start:end]
            else:
                output = x[:, start:end].clone(memory_format=torch.contiguous_format)
            outputs.append(output)
        # No cat/reduction: keep the same per-field outputs as DeepCTR lookup.
        return tuple(outputs)


def prepare_input(model, x):
    """Prepare ablation inputs outside the measured/compiled region."""
    if model.case in ("slice_cast_embedding", "slice_cast", "slice_only", "slice_copy"):
        return x
    columns = tuple(x[:, start:end] for start, end in model.feature_index.values())
    if model.case == "embedding_only":
        return tuple(column.long() for column in columns)
    return columns
