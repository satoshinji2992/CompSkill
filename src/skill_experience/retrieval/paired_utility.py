"""Task-balanced, discordance-only updates from matched skill executions.

The same task must have been executed with each observed candidate. A pair
where both arms succeed or both fail has no relative-utility label. This
module contains no benchmark I/O; held-out labels are never needed to fit.
"""

from __future__ import annotations

from itertools import combinations

import torch
from torch import nn

from skill_experience.retrieval.calibrated_key import CalibratedKey, fit_coordinates


def directional_pairs(candidates: torch.Tensor, success: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return task, winner, loser, and task-normalized weight tensors."""
    if (candidates.ndim != 2 or success.shape != candidates.shape
            or not torch.is_floating_point(success)
            or not torch.isfinite(success).all()
            or not ((success == 0) | (success == 1)).all()):
        raise ValueError("Require matched candidates and binary observed outcomes")
    if (candidates < 0).any() or any(len(set(row)) != len(row)
                                     for row in candidates.detach().cpu().tolist()):
        raise ValueError("Candidate skill IDs must be nonnegative and unique per task")
    by_task: list[list[tuple[int, int]]] = []
    for row_skills, row_success in zip(candidates.tolist(), success.tolist()):
        pairs = []
        for left, right in combinations(range(len(row_skills)), 2):
            if row_success[left] == row_success[right]:
                continue
            winner, loser = (left, right) if row_success[left] > row_success[right] else (right, left)
            pairs.append((row_skills[winner], row_skills[loser]))
        by_task.append(pairs)
    active_tasks = sum(bool(pairs) for pairs in by_task)
    rows: list[tuple[int, int, int, float]] = []
    for task, pairs in enumerate(by_task):
        if pairs:
            rows.extend((task, winner, loser, 1.0 / (active_tasks * len(pairs)))
                        for winner, loser in pairs)
    device = candidates.device
    if not rows:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty.clone(), empty.clone(), torch.empty(0, device=device)
    task, winner, loser, weight = zip(*rows)
    return (torch.tensor(task, dtype=torch.long, device=device),
            torch.tensor(winner, dtype=torch.long, device=device),
            torch.tensor(loser, dtype=torch.long, device=device),
            torch.tensor(weight, dtype=torch.float32, device=device))


def directional_ranking_loss(scores: torch.Tensor, pairs: tuple[torch.Tensor, ...],
                             temperature: float = 0.05) -> torch.Tensor:
    if scores.ndim != 2 or temperature <= 0:
        raise ValueError("Scores must be a matrix and temperature positive")
    task, winner, loser, weight = pairs
    if not len(task):
        return scores.sum() * 0
    if (winner >= scores.shape[1]).any() or (loser >= scores.shape[1]).any():
        raise ValueError("Observed candidate lies outside the scored skill bank")
    gap = (scores[task, winner] - scores[task, loser]) / temperature
    return (nn.functional.softplus(-gap) * weight.to(scores.dtype)).sum()


def fit_bounded_key(
    queries: torch.Tensor,
    base_keys: torch.Tensor,
    candidates: torch.Tensor,
    success: torch.Tensor,
    *,
    mode: str,
    radius: float,
    steps: int,
    learning_rate: float,
    keep: float,
    temperature: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Fit bias, PCA8, or full linear residual as exactly foldable index arrays.

    This is a primitive for a protocol-selected configuration, not a search
    over hyperparameters. The output scores are ``q @ key.T + bias``.
    """
    if mode not in {"bias", "pca8", "linear"}:
        raise ValueError("Only bias, PCA8 and linear conditional keys are supported")
    if (queries.ndim != 2 or base_keys.ndim != 2
            or queries.shape[1] != base_keys.shape[1]
            or candidates.ndim != 2 or len(candidates) != len(queries)
            or steps < 1 or learning_rate <= 0 or keep < 0 or radius <= 0):
        raise ValueError("Invalid paired key fit inputs")
    if candidates.numel() and int(candidates.max()) >= len(base_keys):
        raise ValueError("Observed candidate lies outside the skill bank")
    pairs = directional_pairs(candidates, success)
    support = torch.zeros(len(base_keys), dtype=torch.bool, device=queries.device)
    if len(pairs[0]):
        support[pairs[1]] = True
        support[pairs[2]] = True
    mean, basis = fit_coordinates(queries, mode)
    model = CalibratedKey(mean, basis, support, radius).to(queries.device)
    if len(pairs[0]):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        base = queries @ base_keys.T
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            correction = model(queries)
            ranking = directional_ranking_loss(base + correction, pairs, temperature)
            regularizer = keep * correction.square().mean()
            (ranking + regularizer).backward()
            optimizer.step()
            model.project()
    with torch.no_grad():
        keys, bias = model.folded(base_keys)
    return keys.detach(), bias.detach(), {
        "observed_tasks": len(queries),
        "directional_tasks": int(torch.unique(pairs[0]).numel()),
        "directional_pairs": len(pairs[0]),
        "supported_skills": int(support.sum()),
        "mode": mode,
    }


def direct_pair_prototype(
    queries: torch.Tensor,
    base_keys: torch.Tensor,
    candidates: torch.Tensor,
    success: torch.Tensor,
    *,
    radius: float,
    scale_mode: str = "clip",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Write a signed query mean into keys, optionally filling a shared radius.

    ``clip`` preserves the historical per-skill clipping behavior. With
    ``global_radius``, one common multiplier sends the largest nonzero raw
    residual to ``radius`` while preserving all skill-to-skill magnitudes.
    """
    if (radius <= 0 or scale_mode not in {"clip", "global_radius"}
            or queries.ndim != 2 or base_keys.ndim != 2
            or queries.shape[1] != base_keys.shape[1]):
        raise ValueError("Invalid prototype shape or radius")
    pairs = directional_pairs(candidates, success)
    residual = torch.zeros_like(base_keys)
    global_scale = None
    if len(pairs[0]):
        task, winner, loser, weight = pairs
        changes = queries[task] * weight[:, None].to(queries.dtype)
        residual.index_add_(0, winner, changes)
        residual.index_add_(0, loser, -changes)
    raw_norms = residual.norm(dim=1)
    raw_max_norm = float(raw_norms.max()) if len(raw_norms) else 0.0
    if scale_mode == "clip":
        scale = (radius / raw_norms.clamp_min(radius)).unsqueeze(1)
        residual *= scale
    elif raw_max_norm > 0:
        global_scale = radius / raw_max_norm
        residual *= global_scale
    else:
        global_scale = 0.0
    return base_keys + residual, base_keys.new_zeros(len(base_keys)), {
        "directional_tasks": int(torch.unique(pairs[0]).numel()),
        "directional_pairs": len(pairs[0]),
        "supported_skills": int((residual.norm(dim=1) > 0).sum()),
        "mode": "direct_pair_prototype",
        "scale_mode": scale_mode,
        "raw_max_norm": raw_max_norm,
        "global_scale": global_scale,
        "updated_max_norm": float(residual.norm(dim=1).max()) if len(residual) else 0.0,
    }


def fit_bias_plus_prototype(
    queries: torch.Tensor,
    base_keys: torch.Tensor,
    candidates: torch.Tensor,
    success: torch.Tensor,
    *,
    radius: float,
    bias_fraction: float,
    steps: int,
    learning_rate: float,
    keep: float,
    temperature: float = 0.05,
    prototype_mode: str = "legacy",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Combine two simple updates under one per-skill correction budget.

    The fitted skill bias gets ``bias_fraction * radius`` and the signed query
    prototype gets the remainder. Thus ``||delta_key|| + |bias| <= radius``
    for every skill, matching the
    bound on a conditional key. Both components read the same matched outcomes;
    neither uses a held-out query or result when called within train folds.
    """
    if not (radius > 0 and 0 < bias_fraction < 1
            and prototype_mode in {"legacy", "scaled_then_bias"}):
        raise ValueError("Positive shared radius, interior bias fraction, and valid mode required")
    bias_radius = radius * bias_fraction
    prototype_radius = radius * (1 - bias_fraction)
    if prototype_mode == "legacy":
        _, bias, bias_audit = fit_bounded_key(
            queries, base_keys, candidates, success, mode="bias",
            radius=bias_radius, steps=steps, learning_rate=learning_rate,
            keep=keep, temperature=temperature)
        keys, _, prototype_audit = direct_pair_prototype(
            queries, base_keys, candidates, success, radius=prototype_radius)
    else:
        keys, _, prototype_audit = direct_pair_prototype(
            queries, base_keys, candidates, success, radius=prototype_radius,
            scale_mode="global_radius")
        _, bias, bias_audit = fit_bounded_key(
            queries, keys, candidates, success, mode="bias",
            radius=bias_radius, steps=steps, learning_rate=learning_rate,
            keep=keep, temperature=temperature)
    return keys, bias, {
        "mode": "bias_plus_prototype",
        "prototype_mode": prototype_mode,
        "observed_tasks": len(queries),
        "directional_tasks": bias_audit["directional_tasks"],
        "directional_pairs": bias_audit["directional_pairs"],
        "supported_skills": bias_audit["supported_skills"],
        "shared_radius": radius,
        "bias_fraction": bias_fraction,
        "bias_radius": bias_radius,
        "prototype_radius": prototype_radius,
        "prototype_supported_skills": prototype_audit["supported_skills"],
        "prototype_raw_max_norm": prototype_audit["raw_max_norm"],
        "prototype_global_scale": prototype_audit["global_scale"],
        "prototype_updated_max_norm": prototype_audit["updated_max_norm"],
    }
