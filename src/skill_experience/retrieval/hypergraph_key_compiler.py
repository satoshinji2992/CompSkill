"""Two-stage message passing over matched execution-evidence hyperedges.

Each hyperedge is one observed comparison event, NOT a jointly executed skill
bundle. Member messages retain skill/outcome association. Nonlinear within-edge
attention precedes the existing skill-to-history attention. Only observed signed
query directions can become key updates; no new online graph traversal is added.
"""

from __future__ import annotations

import torch
from torch import nn

from skill_experience.retrieval.evidence_key_compiler import (
    EvidenceKeyCompiler,
    observed_evidence_bank,
)
from skill_experience.retrieval.skill_evidence_graph import SkillEvidenceGraph


def graph_execution_matrix(graph, *, task_ids, queries, skill_refs, excluded_groups,
                           executor_id, environment_id, seed):
    """Reuse graph version/split permissions and export only observed singleton arms."""
    if not isinstance(graph, SkillEvidenceGraph) or not graph.is_frozen:
        raise ValueError("A frozen evidence graph is required")
    if len(set(task_ids)) != len(task_ids) or set(task_ids) != set(queries):
        raise ValueError("Explicit unique query and task identities are required")
    refs = {ref: i for i, ref in enumerate(skill_refs)}
    if len(refs) != len(skill_refs):
        raise ValueError("Duplicated skill version")
    episodes = graph.view(allowed_train_task_ids=task_ids,
        excluded_source_groups=excluded_groups, executor_id=executor_id,
        environment_id=environment_id, skill_versions=skill_refs, seed=seed)
    lookup = {}
    for episode in episodes:
        if episode.task_id in lookup:
            raise ValueError("Multiple contexts or repeats cannot be collapsed into one event")
        lookup[episode.task_id] = episode
    if set(lookup) != set(task_ids):
        raise ValueError("Requested event lacks authorized feedback")
    q = torch.stack([queries[task_id] for task_id in task_ids])
    y = q.new_full((len(q), len(refs)), float("nan"))
    seen = torch.zeros(y.shape, dtype=torch.bool, device=q.device)
    for row, task_id in enumerate(task_ids):
        for arm in lookup[task_id].arms:
            if len(arm.skills) != 1:
                continue  # Bundle outcomes never become individual skill labels.
            col = refs[arm.skills[0]]
            if seen[row, col]:
                raise ValueError("Duplicate physical observation in event")
            y[row, col] = arm.outcome
            seen[row, col] = True
    return {"queries": q, "outcomes": y, "observed": seen,
            "groups": [lookup[task_id].source_group for task_id in task_ids],
            "evidence_ids": [lookup[task_id].evidence_id for task_id in task_ids]}


class HypergraphKeyCompiler(nn.Module):
    """Flat summary / independent pair messages / joint hyperedge messages.

    A lossless graph with relation nodes can implement this same operator. The
    pairwise control is a restricted message model, not every possible GNN.
    """

    def __init__(self, dimension=512, hidden=64, mode="hyperedge", max_delta=.3):
        super().__init__()
        if mode not in ("flat", "pairwise", "hyperedge"):
            raise ValueError("Unknown evidence relation mode")
        self.config = dict(dimension=dimension, hidden=hidden, mode=mode, max_delta=max_delta)
        self.mode = mode
        self.core = EvidenceKeyCompiler(dimension, hidden, "attention", max_delta)
        if mode != "flat":
            self.member = nn.Sequential(nn.Linear(dimension + 2, hidden),
                                        nn.LayerNorm(hidden), nn.SiLU())
            self.edge_read = (nn.MultiheadAttention(hidden, 4, batch_first=True)
                              if mode == "hyperedge" else None)
            self.member_post = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                             nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
            self.merge = nn.Linear(hidden, hidden, bias=False)

    def relation_messages(self, keys, queries, outcomes, observed):
        """Return skill×event messages; never use missing outcome values."""
        clean = outcomes.masked_fill(~observed, 0)
        similarity = queries @ keys.T
        features = torch.cat((keys[None].expand(len(queries), -1, -1),
                              clean[..., None], similarity[..., None]), -1)
        tokens = self.member(features.masked_fill(~observed[..., None], 0))
        mask = observed.clone()
        empty = ~mask.any(-1)
        mask[empty, 0] = True  # One zero placeholder avoids all-masked softmax NaNs.
        tokens = tokens.masked_fill(~observed[..., None], 0)
        if self.edge_read is not None:
            joint, _ = self.edge_read(tokens, tokens, tokens,
                                      key_padding_mask=~mask, need_weights=False)
            tokens = tokens + joint
        messages = self.member_post(tokens).masked_fill(~observed[..., None], 0)
        peers = messages.sum(1, keepdim=True) - messages
        counts = observed.sum(-1, keepdim=True) - observed.to(torch.long)
        peers = peers / counts.clamp_min(1)[..., None]
        peers = peers.masked_fill((counts == 0)[..., None], 0)
        return self.merge(peers).transpose(0, 1)

    def forward(self, keys, queries, outcomes, observed, *, return_details=False):
        if len(queries) == 0:
            # Keep an autograd path for a legitimate zero-feedback episode.
            zero = self.core.gates.weight.sum() * 0
            delta, bias = keys * 0 + zero, keys.new_zeros(len(keys)) + zero
            return (delta, bias, {}) if return_details else (delta, bias)
        evidence, directions, owners = observed_evidence_bank(keys=keys, queries=queries,
                                                             outcomes=outcomes, observed=observed)
        valid = owners >= 0
        if self.mode == "flat":
            return self.core(keys, evidence, directions, valid, return_details=return_details)
        tokens = self.core.evidence(evidence)
        tokens = tokens + self.relation_messages(keys, queries, outcomes, observed)
        tokens = tokens.masked_fill(~valid[..., None], 0)
        tokens = torch.cat((tokens, self.core.null.expand(len(keys), -1, -1)), 1)
        padding = torch.cat((~valid, torch.zeros_like(valid[:, :1])), 1)
        anchor = self.core.skill(keys)
        read, weights = self.core.readout(anchor[:, None], tokens, tokens,
                                         key_padding_mask=padding, need_weights=True)
        read, weights = read[:, 0], weights[:, 0, :-1]
        count = valid.sum(-1, keepdim=True).to(keys.dtype).log1p()
        gates = -torch.expm1(-self.core.gates(torch.cat((anchor, read, count), -1)).relu())
        direction = (directions * weights[..., None]).sum(1)
        delta = self.core.max_delta * gates[:, :1] * direction[:, :-1]
        bias = self.core.max_delta * gates[:, 1] * direction[:, -1]
        scale = self.core.max_delta / (delta.norm(dim=-1) + bias.abs()).clamp_min(
            self.core.max_delta)
        scale = scale * valid.any(-1)
        delta, bias = delta * scale[:, None], bias * scale
        if return_details:
            return delta, bias, {"weights": weights, "gates": gates,
                                 "support_count": valid.sum(-1)}
        return delta, bias
