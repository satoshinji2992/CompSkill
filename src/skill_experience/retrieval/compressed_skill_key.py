"""Recovered FullAnchor embedding-only MLP, preserving original state-dict names.

This is the embedding branch of the original full-anchor use-key model.
Token readers are omitted; no large model or legacy package is imported.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class EmbeddingBlock(nn.Module):
    def __init__(self, width, expansion):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, expansion), nn.SiLU(), nn.Linear(expansion, width)
        )

    def forward(self, q):
        return q + self.mlp(self.norm(q))


class CompressedSkillKey(nn.Module):
    def __init__(self, condition_width=4096, width=512, layers=2, expansion=4096):
        super().__init__()
        self.condition_width, self.width = condition_width, width
        self.condition_projection = nn.Linear(condition_width, width)
        self.condition_norm = nn.LayerNorm(width)
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, width)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.log_scale = nn.Parameter(torch.tensor(math.log(0.1)))
        self.blocks = nn.ModuleList([EmbeddingBlock(width, expansion) for _ in range(layers)])

    def forward(self, condition):
        if (
            condition.ndim != 2
            or condition.shape[1] != self.condition_width
            or condition.requires_grad
            or not torch.isfinite(condition).all()
            or not (condition[:, : self.width].float().norm(dim=-1) > 0).all()
        ):
            raise ValueError("Invalid frozen full conditioning vector")
        q = self.condition_norm(self.condition_projection(F.normalize(condition.float(), dim=-1)))
        for block in self.blocks:
            q = block(q)
        raw = condition[:, : self.width].float()
        delta = self.output_projection(self.output_norm(q)).float()
        return F.normalize(
            raw + raw.norm(dim=-1, keepdim=True) * self.log_scale.exp() * delta, dim=-1
        )
