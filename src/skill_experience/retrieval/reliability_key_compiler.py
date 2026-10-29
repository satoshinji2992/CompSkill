"""Matched reliability calibration, offline relation readout, frozen-bias KeyOpt.

No benchmark I/O. Callers provide TRAIN support filtered by the existing versioned
evidence graph. Graph corrections are scalar; conditional keys are fitted in a
separate stage with the entire calibrated bias detached and frozen.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from skill_experience.retrieval.evidence_key_compiler import (
    expected_utility_loss,
    observed_evidence_bank,
)
from skill_experience.retrieval.hypergraph_key_compiler import HypergraphKeyCompiler
from skill_experience.retrieval.paired_utility import directional_ranking_loss


def _positive(value, name, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (value == 0 and not zero)):
        raise ValueError(f"Invalid {name}")


def validate_evidence(keys, queries, outcomes, observed):
    if (keys.ndim != 2 or queries.ndim != 2 or not all(keys.shape)
            or queries.shape[1] != keys.shape[1]
            or outcomes.shape != (len(queries), len(keys)) or observed.shape != outcomes.shape
            or observed.dtype != torch.bool):
        raise ValueError("Require keys[S,D], queries[N,D], outcomes/mask[N,S]")
    if (keys.dtype not in (torch.float32, torch.float64) or queries.dtype != keys.dtype
            or outcomes.dtype != keys.dtype or any(x.device != keys.device
                for x in (queries, outcomes, observed))
            or not torch.isfinite(keys).all() or not torch.isfinite(queries).all()
            or not torch.isfinite(outcomes[observed]).all()
            or not ((outcomes[observed] == 0) | (outcomes[observed] == 1)).all()):
        raise ValueError("Require aligned finite floating features and observed binary outcomes")
    if any(x.requires_grad for x in (keys, queries, outcomes)):
        raise ValueError("Base features and execution labels must be detached")


def masked_pairs(outcomes: Tensor, observed: Tensor):
    """Reuse existing pair construction, balancing ragged events equally."""
    # One CPU transfer, rather than one validation/synchronization per event.
    # Preserve the established directional_pairs ordering and task weighting.
    from itertools import combinations

    labels = outcomes.detach().cpu().tolist()
    masks = observed.detach().cpu().tolist()
    rows = []
    active = 0
    for task, (values, visible) in enumerate(zip(labels, masks, strict=True)):
        pairs = []
        for left, right in combinations([i for i, seen in enumerate(visible) if seen], 2):
            if values[left] != values[right]:
                winner, loser = (left, right) if values[left] > values[right] else (right, left)
                pairs.append((winner, loser))
        if pairs:
            active += 1
            rows.extend((task, win, lose, 1. / len(pairs)) for win, lose in pairs)
    if not rows:
        empty = torch.empty(0, dtype=torch.long, device=outcomes.device)
        return empty, empty.clone(), empty.clone(), outcomes.new_empty(0)
    task, winner, loser, weight = zip(*rows)
    return (torch.tensor(task, dtype=torch.long, device=outcomes.device),
            torch.tensor(winner, dtype=torch.long, device=outcomes.device),
            torch.tensor(loser, dtype=torch.long, device=outcomes.device),
            torch.tensor(weight, dtype=torch.float32, device=outcomes.device).to(outcomes.dtype) / active)


def evidence_counts(outcomes, observed, pairs):
    hits = torch.zeros_like(observed)
    task, winner, loser, _ = pairs
    hits[task, winner] = True
    hits[task, loser] = True
    return observed.sum(0).to(outcomes.dtype), hits.sum(0).to(outcomes.dtype)


@dataclass(frozen=True)
class ReliabilityPrior:
    bias: Tensor
    observations: Tensor
    comparisons: Tensor
    evidence_sha256: str


def evidence_fingerprint(keys, queries, outcomes, observed):
    digest = hashlib.sha256()
    for value in (keys, queries, outcomes.masked_fill(~observed, 0), observed):
        value = value.detach().contiguous()
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_reliability_prior(keys, queries, outcomes, observed, *, steps=300,
                          learning_rate=.03, radius=.3, keep=.01,
                          temperature=.1, pseudocount=4.) -> tuple[ReliabilityPrior, dict]:
    """Fit matched, task-balanced calibration, not raw success probabilities.

    The semantic score is an offset. Missing arms/singletons/ties create no
    relative label. Low support strengthens shrinkage; candidate selection can
    still bias the evidence, so this is not an unbiased causal ability estimate.
    """
    validate_evidence(keys, queries, outcomes, observed)
    _settings(steps, learning_rate, radius, keep, temperature)
    _positive(pseudocount, "pseudocount")
    pairs = masked_pairs(outcomes, observed)
    observations, comparisons = evidence_counts(outcomes, observed, pairs)
    support = comparisons > 0
    raw = nn.Parameter(keys.new_zeros(len(keys)))
    base = queries @ keys.T
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    initial = float(directional_ranking_loss(base, pairs, temperature))
    for _ in range(steps if len(pairs[0]) and radius > 0 else 0):
        optimizer.zero_grad(set_to_none=True)
        ranking = directional_ranking_loss(base + raw * support, pairs, temperature)
        shrink = pseudocount / (comparisons[support] + pseudocount)
        loss = ranking + keep * (shrink * (raw[support] / radius).square()).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite reliability objective")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            raw.mul_(support)
            raw.clamp_(-radius, radius)
    bias = raw.detach().clone()
    prior = ReliabilityPrior(bias, observations.detach(), comparisons.detach(),
                             evidence_fingerprint(keys, queries, outcomes, observed))
    return prior, {"directional_events": int(pairs[0].unique().numel()),
                   "directional_pairs": len(pairs[0]), "initial_ranking": initial,
                   "final_ranking": float(directional_ranking_loss(base + bias, pairs, temperature)),
                   "max_abs_bias": float(bias.abs().max())}


def _settings(steps, lr, radius, keep, temperature):
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    for val, name, zero in [(lr, "learning_rate", False), (radius, "radius", True),
                            (keep, "keep", True), (temperature, "temperature", False)]:
        _positive(val, name, zero=zero)


class ReliabilityGraphCompiler(nn.Module):
    """Preserve a supplied reliability prior and learn a bounded scalar correction.

    Modes share the same visible evidence. ``factor`` uses a full event node,
    not a lossy pair projection; it is a genuine competing relation model.
    Hyperedge attention is not assumed more expressive than general factor GNNs.
    The zero-initialized head makes initial deployment exactly the prior. Relation
    weights receive gradients after the head starts moving, not on the first step.
    """

    def __init__(self, dimension=512, hidden=64, mode="hyperedge", radius=.1, pseudocount=4.):
        super().__init__()
        if mode not in {"flat", "pairwise", "factor", "hyperedge"}:
            raise ValueError("Unknown relation mode")
        _positive(radius, "radius", zero=True)
        _positive(pseudocount, "pseudocount")
        backend = HypergraphKeyCompiler(dimension, hidden,
            "pairwise" if mode == "factor" else mode)
        # Retain only used blocks; old vector/bias gates are deliberately absent.
        self.skill, self.evidence = backend.core.skill, backend.core.evidence
        self.readout, self.null = backend.core.readout, backend.core.null
        if mode != "flat":
            self.member, self.member_post, self.merge = backend.member, backend.member_post, backend.merge
            self.edge_read = backend.edge_read
        if mode == "factor":
            self.event_node = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                            nn.Linear(hidden, hidden))
        self.prior_features = nn.Linear(3, hidden)
        self.head = nn.Linear(2 * hidden + 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.mode, self.radius, self.pseudocount = mode, radius, pseudocount
        self.config = dict(dimension=dimension, hidden=hidden, mode=mode,
                           radius=radius, pseudocount=pseudocount)

    def _relations(self, keys, queries, outcomes, observed):
        if self.mode != "factor":
            return HypergraphKeyCompiler.relation_messages(self, keys, queries, outcomes, observed)
        clean = outcomes.masked_fill(~observed, 0)
        features = torch.cat((keys[None].expand(len(queries), -1, -1),
                              clean[..., None], (queries @ keys.T)[..., None]), -1)
        tokens = self.member(features.masked_fill(~observed[..., None], 0))
        tokens = tokens.masked_fill(~observed[..., None], 0)
        event = tokens.sum(1) / observed.sum(1).clamp_min(1)[:, None]
        # Nonlinear skill/outcome tokens -> event node -> every original member.
        messages = self.member_post(tokens + self.event_node(event)[:, None])
        return self.merge(messages.masked_fill(~observed[..., None], 0)).transpose(0, 1)

    def forward(self, keys, queries, outcomes, observed, prior, *, return_details=False):
        validate_evidence(keys, queries, outcomes, observed)
        if not isinstance(prior, ReliabilityPrior):
            raise ValueError("Explicit fitted reliability prior required")
        if prior.evidence_sha256 != evidence_fingerprint(keys, queries, outcomes, observed):
            raise ValueError("Prior is bound to different evidence or skill order")
        for x in (prior.bias, prior.observations, prior.comparisons):
            if (x.shape != (len(keys),) or x.dtype != keys.dtype or x.device != keys.device
                    or x.requires_grad or not torch.isfinite(x).all()):
                raise ValueError("Prior must be detached, finite and skill-aligned")
        if (prior.observations < 0).any() or (prior.comparisons < 0).any():
            raise ValueError("Negative prior support")
        pairs = masked_pairs(outcomes, observed)
        counts, comparisons = evidence_counts(outcomes, observed, pairs)
        if not torch.equal(counts, prior.observations) or not torch.equal(comparisons, prior.comparisons):
            raise ValueError("Prior support differs from visible graph evidence")
        stats = torch.stack((prior.bias, counts.log1p(), comparisons.log1p()), -1)
        anchor = self.skill(keys) + self.prior_features(stats)
        if not len(queries):
            delta = prior.bias * 0 + self.head.weight.sum() * 0
            details = {"correction": delta, "weights": keys.new_zeros((len(keys), 0))}
            return (prior.bias + delta, details) if return_details else prior.bias + delta
        evidence, _, owners = observed_evidence_bank(queries, keys, outcomes, observed)
        valid = owners >= 0
        tokens = self.evidence(evidence)
        if self.mode != "flat":
            tokens = tokens + self._relations(keys, queries, outcomes, observed)
        tokens = tokens.masked_fill(~valid[..., None], 0)
        tokens = torch.cat((tokens, self.null.expand(len(keys), -1, -1)), 1)
        padding = torch.cat((~valid, torch.zeros((len(keys), 1), dtype=torch.bool,
                                                device=keys.device)), 1)
        readout, weights = self.readout(anchor[:, None], tokens, tokens,
                                        key_padding_mask=padding, need_weights=True)
        raw = self.head(torch.cat((anchor, readout[:, 0], stats), -1)).squeeze(-1)
        confidence = comparisons / (comparisons + self.pseudocount)
        delta = self.radius * raw.tanh() * confidence
        bias = prior.bias + delta
        details = {"correction": delta, "weights": weights[:, 0, :-1],
                   "support_count": counts, "comparison_count": comparisons}
        return (bias, details) if return_details else bias


def require_disjoint_groups(support_ids, target_ids, support_groups, target_groups):
    if (len(support_ids) != len(support_groups) or len(target_ids) != len(target_groups)
            or len(set(support_ids)) != len(support_ids) or len(set(target_ids)) != len(target_ids)
            or not support_ids or not target_ids
            or any(not isinstance(x, str) or not x for seq in
                   (support_ids, target_ids, support_groups, target_groups) for x in seq)):
        raise ValueError("Explicit unique events and source groups required")
    if set(support_ids) & set(target_ids) or set(support_groups) & set(target_groups):
        raise ValueError("Support and target events/source groups overlap")


def compiler_episode_loss(model, keys, support_q, support_y, support_seen, prior,
                          target_q, target_y, target_seen, *, support_ids, target_ids,
                          support_groups, target_groups, keep=.01, temperature=.1):
    """Shared rule supervision; targets cannot enter compiled memory or its prior."""
    require_disjoint_groups(support_ids, target_ids, support_groups, target_groups)
    if len(support_ids) != len(support_q) or len(target_ids) != len(target_q):
        raise ValueError("Event metadata does not align with evidence")
    validate_evidence(keys, target_q, target_y, target_seen)
    _positive(keep, "keep", zero=True)
    bias, details = model(keys, support_q, support_y, support_seen, prior, return_details=True)
    utility, parts = expected_utility_loss(target_q @ keys.T + bias, target_y, target_seen,
                                          temperature=temperature, pair_weight=0.)
    penalty = details["correction"].square().mean()
    return utility + keep * penalty, {**parts, "graph_penalty": penalty}, bias


@dataclass(frozen=True)
class FrozenReliabilityIndex:
    """Exact affine index. Metadata/checkpoint hashing is the runner's responsibility."""
    keys: Tensor
    bias: Tensor
    reliability_bias: Tensor
    training_mean: Tensor
    delta: Tensor

    def score(self, queries):
        if (queries.ndim != 2 or queries.shape[1] != self.keys.shape[1]
                or queries.device != self.keys.device or queries.dtype != self.keys.dtype
                or not torch.isfinite(queries).all()):
            raise ValueError("Invalid deployment queries")
        return queries @ self.keys.T + self.bias

    def augmented_keys(self):
        return torch.cat((self.keys, self.bias[:, None]), -1)


def fit_frozen_bias_keyopt(keys, queries, outcomes, observed, reliability_bias, *,
                           steps=500, learning_rate=.03, radius=.1, keep=.01,
                           temperature=.1) -> tuple[FrozenReliabilityIndex, dict]:
    """Fit only a centered vector residual, preserving the entire scalar anchor.

    This does not jointly retrain the graph. A zero radius is exact no-update.
    Unknown target arms are masked, never assigned failure. No row normalization
    is applied after folding, and there is no second free learned bias.
    """
    validate_evidence(keys, queries, outcomes, observed)
    _settings(steps, learning_rate, radius, keep, temperature)
    if (reliability_bias.shape != (len(keys),) or reliability_bias.device != keys.device
            or reliability_bias.dtype != keys.dtype or not torch.isfinite(reliability_bias).all()):
        raise ValueError("Invalid frozen reliability bias")
    anchor = reliability_bias.detach().clone()
    mean = queries.mean(0) if len(queries) else keys.new_zeros(keys.shape[1])
    pairs = masked_pairs(outcomes, observed)
    _, counts = evidence_counts(outcomes, observed, pairs)
    support = counts > 0
    # Empty rows have no supervised choice and cannot enter softmax.
    active = observed.any(1)
    base = queries[active] @ keys.T + anchor
    centered = queries[active] - mean
    residual = nn.Parameter(torch.zeros_like(keys))
    optimizer = torch.optim.Adam([residual], lr=learning_rate)
    losses = []
    for _ in range(steps if len(pairs[0]) and radius > 0 else 0):
        optimizer.zero_grad(set_to_none=True)
        utility, _ = expected_utility_loss(base + centered @ residual.T,
            outcomes[active], observed[active], temperature=temperature, pair_weight=0.)
        penalty = (residual[support] / radius).square().sum(1).mean()
        loss = utility + keep * penalty
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite conditional objective")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            residual.mul_(support[:, None])
            residual.mul_((radius / residual.norm(dim=1).clamp_min(radius))[:, None])
        losses.append(float(loss.detach()))
    delta = residual.detach().clone()
    index = FrozenReliabilityIndex((keys + delta).detach(), (anchor - delta @ mean).detach(),
                                    anchor, mean.detach(), delta)
    return index, {"steps_completed": len(losses), "initial_loss": losses[0] if losses else None,
                   "last_pre_step_loss": losses[-1] if losses else None,
                   "max_delta_norm": float(delta.norm(dim=1).max()),
                   "directional_events": int(pairs[0].unique().numel())}
