"""Offline, lesson-conditioned compilation of bounded retrieval-key updates."""

import torch
from torch import nn


class LessonKeyCompiler(nn.Module):
    """Matched evidence encoding for mean and skill-conditioned attention.

    Evidence contains historical support outcomes. Target-task outcomes must
    never be included in the valid mask; callers use support_mask for this.
    """

    def __init__(self, dimension=512, lesson_dimension=768, hidden=128,
                 mode="attention", max_delta=.1):
        super().__init__()
        if mode not in ("mean", "attention"):
            raise ValueError("Unknown evidence aggregation")
        self.config = dict(dimension=dimension, lesson_dimension=lesson_dimension,
                           hidden=hidden, mode=mode, max_delta=max_delta)
        self.mode, self.max_delta = mode, max_delta
        self.skill = nn.Linear(dimension, hidden)
        width = 2 * dimension + 2 * lesson_dimension + 8
        self.evidence = nn.Sequential(nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.readout = (nn.MultiheadAttention(hidden, 4, batch_first=True)
                        if mode == "attention" else nn.Sequential(
                            nn.Linear(hidden, 2 * hidden), nn.SiLU(), nn.Linear(2 * hidden, hidden)))
        self.head = nn.Linear(2 * hidden + 1, dimension + 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, keys, evidence, valid):
        if valid.dtype != torch.bool or evidence.shape[:2] != valid.shape:
            raise ValueError("Evidence mask differs")
        if len(keys) != len(evidence) or not evidence.shape[1]:
            raise ValueError("Provide one masked placeholder for empty skills")
        supported = valid.any(-1)
        mask = valid.clone()
        mask[~supported, 0] = True
        # Clear masked records before encoding: even NaNs or arbitrary held-out
        # record changes cannot influence valid values through the readout.
        tokens = self.evidence(evidence.masked_fill(~valid[..., None], 0))
        anchor = self.skill(keys)
        if self.mode == "attention":
            read, _ = self.readout(anchor[:, None], tokens, tokens,
                                   key_padding_mask=~mask, need_weights=False)
            read = read[:, 0]
        else:
            read = (tokens * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            read = self.readout(read)
        count = valid.sum(-1, keepdim=True).to(keys.dtype).log1p()
        raw = self.head(torch.cat((anchor, read, count), -1))
        norm = raw[:, :-1].norm(dim=-1) + raw[:, -1].abs()
        raw = raw * (self.max_delta / norm.clamp_min(self.max_delta))[:, None]
        raw = raw * supported[:, None]
        return raw[:, :-1], raw[:, -1]


def lesson_evidence_bank(queries, keys, pairs, with_lesson, without_lesson,
                         lesson_vectors, condition_vectors, present):
    """Keep both ties and differences as evidence; IDs track source tasks.

    lesson/condition vectors and present are [tasks, 2, ...]. Success bits are
    observed outcomes, not estimates of deterministic causal uplift.
    """
    per_skill = [[] for _ in keys]
    for task in range(len(queries)):
        q = queries[task]
        for own_side in (0, 1):
            other_side = 1 - own_side
            own, other = (int(pairs[task, own_side]), int(pairs[task, other_side]))
            yh, yo = with_lesson[task].float(), without_lesson[task].float()
            features = torch.stack((yh[own_side], yh[other_side], yo[own_side], yo[other_side],
                yh[own_side] - yo[own_side], yh[other_side] - yo[other_side],
                q @ (keys[own] - keys[other]), present[task, own_side].float()))
            token = torch.cat((q, keys[other], lesson_vectors[task, own_side],
                               condition_vectors[task, own_side], features))
            per_skill[own].append((token, task))
    width = max(1, max(map(len, per_skill)))
    features = 2 * keys.shape[1] + 2 * lesson_vectors.shape[-1] + 8
    evidence = keys.new_zeros((len(keys), width, features))
    owners = torch.full((len(keys), width), -1, dtype=torch.long, device=keys.device)
    for sid, rows in enumerate(per_skill):
        for n, (token, task) in enumerate(rows):
            evidence[sid, n], owners[sid, n] = token, task
    return evidence, owners
