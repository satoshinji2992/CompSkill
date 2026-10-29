"""The established reliability/KeyOpt recipe with frozen cross-encoder offsets.

No benchmark I/O. Unknown candidates never enter a training comparison. This
uses the same losses, support mask, shrinkage, centering and optimizer settings
as reliability_key_compiler; only its semantic score is supplied explicitly.
"""
from __future__ import annotations

import torch
from torch import nn
from skill_experience.retrieval.reliability_key_compiler import (
    masked_pairs, evidence_counts, _settings,
)
from skill_experience.retrieval.paired_utility import directional_ranking_loss
from skill_experience.retrieval.evidence_key_compiler import expected_utility_loss


def validate(scores, queries, outcomes, observed):
    if scores.ndim != 2 or scores.shape != outcomes.shape or observed.shape != scores.shape:
        raise ValueError('Scores/outcomes/mask must be aligned')
    if queries.ndim != 2 or len(queries) != len(scores) or observed.dtype != torch.bool:
        raise ValueError('Queries/mask invalid')
    if any(x.requires_grad for x in (scores, queries, outcomes)):
        raise ValueError('Frozen inputs required')
    if not torch.isfinite(scores[observed]).all() or not torch.isfinite(queries).all():
        raise ValueError('Nonfinite known scores or queries')
    if not ((outcomes[observed] == 0) | (outcomes[observed] == 1)).all():
        raise ValueError('Observed outcomes must be binary')


def fit_bias(scores, queries, outcomes, observed, *, steps=300, learning_rate=.03,
             radius=.3, keep=.01, temperature=.1, pseudocount=4.):
    validate(scores, queries, outcomes, observed)
    _settings(steps, learning_rate, radius, keep, temperature)
    base = scores.masked_fill(~observed, 0)
    pairs = masked_pairs(outcomes, observed)
    _, counts = evidence_counts(outcomes, observed, pairs)
    support = counts > 0
    bias = nn.Parameter(queries.new_zeros(scores.shape[1]))
    optimizer = torch.optim.Adam([bias], lr=learning_rate)
    initial = float(directional_ranking_loss(base, pairs, temperature))
    for _ in range(steps if len(pairs[0]) and radius else 0):
        optimizer.zero_grad(set_to_none=True)
        shrink = pseudocount / (counts[support] + pseudocount)
        loss = directional_ranking_loss(base + bias * support, pairs, temperature)
        loss = loss + keep * (shrink * (bias[support] / radius).square()).mean()
        if not torch.isfinite(loss): raise FloatingPointError('Bias loss')
        loss.backward(); optimizer.step()
        with torch.no_grad(): bias.mul_(support).clamp_(-radius, radius)
    return bias.detach(), dict(initial_ranking=initial,
        final_ranking=float(directional_ranking_loss(base+bias, pairs, temperature)),
        max_abs_bias=float(bias.detach().abs().max()))


def fit_residual(scores, queries, outcomes, observed, bias, *, steps=500,
                 learning_rate=.03, radius=.1, keep=.01, temperature=.1):
    validate(scores, queries, outcomes, observed)
    _settings(steps, learning_rate, radius, keep, temperature)
    pairs = masked_pairs(outcomes, observed)
    _, counts = evidence_counts(outcomes, observed, pairs)
    support = counts > 0; active = observed.any(1)
    mean = queries.mean(0)
    centered = queries[active] - mean
    base = scores.masked_fill(~observed, 0)[active] + bias.detach()
    delta = nn.Parameter(queries.new_zeros((scores.shape[1], queries.shape[1])))
    optimizer = torch.optim.Adam([delta], lr=learning_rate)
    losses = []
    for _ in range(steps if len(pairs[0]) and radius else 0):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = expected_utility_loss(base+centered@delta.T,
            outcomes[active], observed[active], temperature=temperature, pair_weight=0.)
        loss = loss + keep*(delta[support]/radius).square().sum(1).mean()
        if not torch.isfinite(loss): raise FloatingPointError('Residual loss')
        loss.backward(); optimizer.step()
        with torch.no_grad():
            delta.mul_(support[:, None])
            delta.mul_((radius/delta.norm(dim=1).clamp_min(radius))[:, None])
        losses.append(float(loss.detach()))
    result = dict(delta=delta.detach(), mean=mean.detach(), bias=bias.detach())
    return result, dict(steps_completed=len(losses), initial_loss=losses[0] if losses else None,
        last_pre_step_loss=losses[-1] if losses else None,
        max_delta_norm=float(delta.detach().norm(dim=1).max()))


def score_adjustment(queries, fitted):
    return fitted['bias'] + (queries-fitted['mean'])@fitted['delta'].T
