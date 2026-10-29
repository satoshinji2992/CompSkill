"""Shared execution-evidence compiler with matched mean/attention readouts.

The compiler runs during an offline update. It emits bounded linear key/bias
corrections; no experience attention is required during task retrieval.
"""

import torch
from torch import nn


class MemoryKeyCompiler(nn.Module):
    def __init__(self, dimension=512, hidden=64, mode="attention", max_delta=.1):
        super().__init__()
        if mode not in ("mean", "attention"):
            raise ValueError("Expected matched mean or attention readout")
        self.config = dict(dimension=dimension, hidden=hidden, mode=mode, max_delta=max_delta)
        self.mode, self.max_delta = mode, max_delta
        self.skill = nn.Linear(dimension, hidden)
        self.evidence = nn.Sequential(nn.Linear(2 * dimension + 2, hidden),
                                      nn.LayerNorm(hidden), nn.SiLU())
        self.readout = (nn.MultiheadAttention(hidden, 4, batch_first=True) if mode == "attention"
                        else nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.SiLU(),
                                           nn.Linear(2 * hidden, hidden)))
        self.head = nn.Linear(2 * hidden, dimension + 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, keys, evidence, valid):
        """evidence[S,N,2D+2], valid[S,N]; all-false skills keep their base key."""
        if evidence.shape[:2] != valid.shape or len(keys) != len(evidence):
            raise ValueError("Mismatched skill/evidence shapes")
        if not evidence.shape[1]:
            raise ValueError("Provide a masked placeholder for an empty evidence bank")
        supported = valid.any(-1)
        mask = valid.clone()
        mask[~supported, 0] = True  # MHA cannot softmax over an entirely masked row.
        tokens = self.evidence(evidence)
        anchor = self.skill(keys)
        if self.mode == "attention":
            read, _ = self.readout(anchor[:, None, :], tokens, tokens,
                                   key_padding_mask=~mask, need_weights=False)
            read = read[:, 0]
        else:
            read = (tokens * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            read = self.readout(read)
        raw = self.head(torch.cat((anchor, read), -1))
        norm = raw[:, :-1].norm(dim=-1) + raw[:, -1].abs()
        raw = raw * (self.max_delta / norm.clamp_min(self.max_delta))[:, None]
        raw = raw * supported[:, None]
        return raw[:, :-1], raw[:, -1]


def evidence_bank(queries, keys, records):
    """Make signed paired evidence and retain task IDs for support/target isolation.

    records contain task_index, left/right skill indices and left/right success.
    Equal outcomes provide no positive/negative preference token.
    """
    per_skill = [[] for _ in keys]
    for r in records:
        if r["left_success"] == r["right_success"]:
            continue
        q = queries[r["task_index"]]
        for own, other, sign in ((r["left"], r["right"], 1 if r["left_success"] else -1),
                                  (r["right"], r["left"], 1 if r["right_success"] else -1)):
            gap = q @ (keys[own] - keys[other])
            extra = torch.stack((q.new_tensor(sign), gap))
            per_skill[own].append((torch.cat((q, keys[other], extra)), r["task_index"]))
    width = max(1, max(map(len, per_skill)))
    values = keys.new_zeros((len(keys), width, 2 * keys.shape[1] + 2))
    owners = torch.full((len(keys), width), -1, dtype=torch.long, device=keys.device)
    for skill, tokens in enumerate(per_skill):
        for i, (token, task) in enumerate(tokens):
            values[skill, i], owners[skill, i] = token, task
    return values, owners


def support_mask(owners, support_ids, target_ids):
    support = torch.as_tensor(support_ids, device=owners.device)
    target = torch.as_tensor(target_ids, device=owners.device)
    if torch.isin(support, target).any():
        raise ValueError("Compiler support and target tasks must be disjoint")
    return torch.isin(owners, support) & (owners >= 0)
