"""Shared update rules constrained to observed execution-pair directions."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from skill_experience.retrieval.lesson_compiler import LessonKeyCompiler


class EvidenceKeyCompiler(nn.Module):
    """Compile signed query directions, with a zero-valued null token."""

    def __init__(self, dimension=512, hidden=64, mode="attention", max_delta=.3):
        super().__init__()
        if mode not in ("mean", "attention") or dimension < 1 or hidden % 4:
            raise ValueError("Invalid matched compiler configuration")
        if not 0 < max_delta <= 1:
            raise ValueError("max_delta must be in (0, 1]")
        self.config = dict(dimension=dimension, hidden=hidden, mode=mode,
                           max_delta=max_delta)
        self.mode, self.max_delta = mode, float(max_delta)
        # Reuse established encoders/aggregation, discarding its free-vector head.
        previous = LessonKeyCompiler(dimension, 0, hidden, mode, max_delta)
        self.skill, self.evidence, self.readout = (
            previous.skill, previous.evidence, previous.readout)
        self.null = nn.Parameter(torch.zeros(1, 1, hidden))
        self.gates = nn.Linear(2 * hidden + 1, 2)
        nn.init.zeros_(self.gates.weight)
        nn.init.constant_(self.gates.bias, .05)

    def forward(self, keys: Tensor, evidence: Tensor, directions: Tensor,
                valid: Tensor, *, return_details=False):
        """Directions[S,N,D+1] are signed unit queries followed by their sign.

        Nonnegative evidence weights cannot reverse an observed preference.
        Ties may influence confidence but their direction is exactly zero.
        Caller controls support; masked records are cleared before every read.
        """
        dimension = self.config["dimension"]
        if (valid.dtype != torch.bool or evidence.shape[:2] != valid.shape
                or directions.shape != (*valid.shape, dimension + 1)
                or keys.shape != (len(evidence), dimension)
                or evidence.shape[-1] != 2 * dimension + 8 or not evidence.shape[1]):
            raise ValueError("Mismatched evidence/compiler shapes")
        evidence = evidence.masked_fill(~valid[..., None], 0)
        directions = directions.masked_fill(~valid[..., None], 0)
        if not (torch.isfinite(keys).all() and torch.isfinite(evidence).all()
                and torch.isfinite(directions).all()):
            raise ValueError("Nonfinite visible evidence")
        tokens = self.evidence(evidence)
        anchor = self.skill(keys)
        supported = valid.any(-1)
        if self.mode == "attention":
            tokens = torch.cat((tokens, self.null.expand(len(keys), -1, -1)), 1)
            padding = torch.cat((~valid, torch.zeros_like(valid[:, :1])), 1)
            read, weights = self.readout(anchor[:, None], tokens, tokens,
                                        key_padding_mask=padding, need_weights=True)
            read, weights = read[:, 0], weights[:, 0, :-1]
        else:
            weights = valid.to(keys.dtype) / valid.sum(-1, keepdim=True).clamp_min(1)
            read = self.readout((tokens * weights[..., None]).sum(1))
        count = valid.sum(-1, keepdim=True).to(keys.dtype).log1p()
        # Nonpositive logits mean exactly no update. Start in the active region.
        gates = -torch.expm1(-self.gates(torch.cat((anchor, read, count), -1)).relu())
        average = (directions * weights[..., None]).sum(1)
        delta = self.max_delta * gates[:, :1] * average[:, :-1]
        bias = self.max_delta * gates[:, 1] * average[:, -1]
        bound = delta.norm(dim=-1) + bias.abs()
        scale = self.max_delta / bound.clamp_min(self.max_delta)
        delta = delta * (scale * supported)[:, None]
        bias = bias * scale * supported
        if return_details:
            return delta, bias, {"weights": weights, "gates": gates,
                                 "support_count": valid.sum(-1)}
        return delta, bias


def observed_evidence_bank(queries: Tensor, keys: Tensor, outcomes: Tensor,
                           observed: Tensor):
    """One token per observed task/skill, regardless of number of opponents.

    Only candidates physically executed under the same task/context/executor/
    seed may occupy a row. Source adapters enforce those identities. Unknown
    cells never become failures. Sole observations or ties give no direction.
    Tokens carry no task or skill identity feature.
    """
    if (queries.ndim != 2 or keys.ndim != 2 or queries.shape[1] != keys.shape[1]
            or outcomes.shape != (len(queries), len(keys))
            or observed.shape != outcomes.shape or observed.dtype != torch.bool):
        raise ValueError("Malformed observed execution matrix")
    if (not torch.isfinite(queries).all() or not torch.isfinite(keys).all()
            or not torch.isfinite(outcomes[observed]).all()
            or not ((outcomes[observed] == 0) | (outcomes[observed] == 1)).all()):
        raise ValueError("Expected finite queries/keys and observed binary success")
    dimension = keys.shape[1]
    evidence = keys.new_zeros((len(keys), max(1, len(queries)), 2 * dimension + 8))
    directions = keys.new_zeros((len(keys), max(1, len(queries)), dimension + 1))
    owners = torch.full(evidence.shape[:2], -1, dtype=torch.long, device=keys.device)
    normalized = torch.nn.functional.normalize(queries, dim=-1)
    clean = outcomes.masked_fill(~observed, 0)
    for own in range(len(keys)):
        others = observed.clone()
        others[:, own] = False
        count = others.sum(-1)
        mean_key = others.to(keys.dtype) @ keys / count.clamp_min(1)[:, None]
        other_y = (clean * others).sum(-1) / count.clamp_min(1)
        sign = (clean[:, own] - other_y) * (count > 0)
        gap = (queries * (keys[own] - mean_key)).sum(-1)
        # Match lesson_evidence_bank width, with no generated lesson vectors.
        features = torch.stack((clean[:, own], other_y, clean[:, own], other_y,
                                sign, count.to(keys.dtype).log1p(), gap,
                                (count > 0).to(keys.dtype)), -1)
        evidence[own, :len(queries)] = torch.cat((queries, mean_key, features), -1)
        directions[own, :len(queries), :-1] = normalized * sign[:, None]
        directions[own, :len(queries), -1] = sign
        owners[own, :len(queries)] = torch.where(
            observed[:, own], torch.arange(len(queries), device=keys.device), -1)
    evidence = evidence.masked_fill((owners < 0)[..., None], 0)
    directions = directions.masked_fill((owners < 0)[..., None], 0)
    return evidence, directions, owners


def expected_utility_loss(scores: Tensor, outcomes: Tensor, observed: Tensor,
                          *, temperature=.1, pair_weight=.1):
    """Expected observed execution utility plus discordant-pair ranking only."""
    if (scores.shape != outcomes.shape or observed.shape != scores.shape
            or observed.dtype != torch.bool or not observed.any(-1).all()
            or temperature <= 0):
        raise ValueError("Invalid target observed-candidate mask")
    clean = outcomes.masked_fill(~observed, 0)
    if (not torch.isfinite(scores[observed]).all()
            or not torch.isfinite(clean).all()):
        raise ValueError("Nonfinite visible targets")
    logits = scores.masked_fill(~observed, -torch.inf) / temperature
    utility = -(logits.softmax(-1) * clean).sum(-1).mean()
    differences = scores[:, :, None] - scores[:, None, :]
    wins = (observed[:, :, None] & observed[:, None, :]
            & (clean[:, :, None] > clean[:, None, :]))
    pair = (torch.nn.functional.softplus(-differences[wins] / temperature).mean()
            if wins.any() else scores[observed].sum() * 0)
    return utility + pair_weight * pair, {"negative_utility": utility, "pair": pair}
