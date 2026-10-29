"""Train-only centered key corrections and task-balanced observed-pair losses."""

import hashlib
from collections import Counter, defaultdict
from itertools import combinations

import torch
from torch import nn


def grouped_folds(records, folds=3):
    """Hash tasks within each family; every state of a task stays in its fold."""
    if folds < 2:
        raise ValueError("At least two folds required")
    tasks = {}
    for row in records:
        task, family = row["task_id"], row["family"]
        if task in tasks and tasks[task] != family:
            raise ValueError("Task belongs to multiple families")
        tasks[task] = family
    families = defaultdict(list)
    for task, family in tasks.items():
        families[family].append(task)
    assignment = {}
    for offset, (family, members) in enumerate(sorted(families.items())):
        ordered = sorted(members, key=lambda task: (hashlib.sha256(task.encode()).hexdigest(), task))
        for index, task in enumerate(ordered):
            assignment[task] = (index + offset) % folds
    if set(assignment.values()) != set(range(folds)):
        raise ValueError("Empty training fold")
    return assignment


def fit_coordinates(train_queries, mode):
    """The caller supplies training rows only; PCA never sees held-out queries."""
    if mode not in ("bias", "linear", "pca8") or train_queries.ndim != 2:
        raise ValueError("Invalid coordinate configuration")
    if not len(train_queries) or not torch.isfinite(train_queries).all():
        raise ValueError("Finite training queries required")
    mean = train_queries.mean(0)
    dimension = train_queries.shape[1]
    if mode == "bias":
        basis = train_queries.new_zeros((dimension, 0))
    elif mode == "linear":
        basis = torch.eye(dimension, dtype=train_queries.dtype, device=train_queries.device)
    else:
        _, _, vectors = torch.linalg.svd(train_queries - mean, full_matrices=False)
        basis = vectors[:min(8, len(vectors))].T.contiguous()
        # Fix SVD sign ambiguity without looking at any labels or held-out states.
        pivots = basis.abs().argmax(0)
        signs = basis[pivots, torch.arange(basis.shape[1], device=basis.device)].sign()
        basis = basis * signs.where(signs != 0, torch.ones_like(signs))
    return mean, basis


class CalibratedKey(nn.Module):
    """bias + (q - mean) W, bounded after exact folding into the original keys."""

    def __init__(self, mean, basis, support, radius):
        super().__init__()
        if (mean.ndim != 1 or basis.ndim != 2 or basis.shape[0] != len(mean)
                or radius <= 0 or not torch.isfinite(mean).all()
                or not torch.isfinite(basis).all()):
            raise ValueError("Invalid centered key configuration")
        self.radius = float(radius)
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("basis", basis.detach().clone())
        self.register_buffer("support", torch.as_tensor(support, dtype=torch.bool).clone())
        if self.support.ndim != 1 or not len(self.support):
            raise ValueError("One support flag per skill required")
        self.coefficients = nn.Parameter(mean.new_zeros((len(self.support), basis.shape[1])))
        self.bias = nn.Parameter(mean.new_zeros(len(self.support)))

    def weights(self):
        return (self.coefficients @ self.basis.T) * self.support[:, None]

    def forward(self, queries):
        return ((queries - self.mean) @ self.basis @ self.coefficients.T + self.bias) * self.support

    def folded(self, base_keys):
        weights = self.weights()
        return base_keys + weights, self.bias * self.support - weights @ self.mean

    @torch.no_grad()
    def project(self):
        weights = self.weights()
        folded_bias = self.bias * self.support - weights @ self.mean
        magnitude = weights.norm(dim=1) + folded_bias.abs()
        scale = (self.radius / magnitude.clamp_min(self.radius)) * self.support
        self.coefficients.mul_(scale[:, None])
        self.bias.mul_(scale)


def utilities(success, progress, objective):
    if objective not in ("success_only", "success_progress"):
        raise ValueError("Unknown objective")
    if (success.shape != progress.shape or not torch.isfinite(progress).all()
            or (progress.abs() > 1).any()
            or not ((success == 0) | (success == 1)).all()):
        raise ValueError("Binary success and public progress in [-1, 1] required")
    return success.float() + (0.2 * progress if objective == "success_progress" else 0)


def directional_support(candidates, utility, skills):
    """Only skills in a directional, actually observed pair receive updates."""
    support = torch.zeros(skills, dtype=torch.bool, device=candidates.device)
    for left, right in combinations(range(candidates.shape[1]), 2):
        active = (utility[:, left] - utility[:, right]).abs() > 1e-8
        support[candidates[active, left]] = True
        support[candidates[active, right]] = True
    return support


def state_weights(task_ids, *, device=None, dtype=torch.float32):
    counts = Counter(task_ids)
    if not counts:
        raise ValueError("No tasks")
    return torch.tensor([1 / (len(counts) * counts[t]) for t in task_ids],
                        device=device, dtype=dtype)


def pair_loss(delta, base_scores, candidates, utility, task_ids, *, validate=True):
    """Unknown candidates have no labels; keep regularization covers the full bank.

    Weighted BCE is divided by the number of active pairs, never their weight sum,
    so the 0.2 public-progress auxiliary cannot become a full-weight success label.
    """
    if (delta.shape != base_scores.shape or candidates.ndim != 2
            or candidates.shape != utility.shape or len(delta) != len(task_ids)
            or len(delta) != len(candidates)):
        raise ValueError("Mismatched loss inputs")
    if validate:
        if candidates.shape[1] < 2 or (candidates < 0).any() or (candidates >= delta.shape[1]).any():
            raise ValueError("Invalid observed candidates")
        if any(len(set(row)) != len(row) for row in candidates.detach().cpu().tolist()):
            raise ValueError("Duplicate observed candidate")
    indices = torch.tensor(list(combinations(range(candidates.shape[1]), 2)),
                           device=delta.device, dtype=torch.long)
    observed_delta = delta.gather(1, candidates)
    observed = (base_scores + delta).gather(1, candidates)
    difference = utility[:, indices[:, 0]] - utility[:, indices[:, 1]]
    active = difference.abs() > 1e-8
    gap = (observed[:, indices[:, 0]] - observed[:, indices[:, 1]]) / 0.05
    bce = nn.functional.binary_cross_entropy_with_logits(
        gap, (difference > 0).to(gap.dtype), reduction="none")
    ranking = (bce * difference.abs() * active).sum(1) / active.sum(1).clamp_min(1)
    correction_gap = (observed_delta[:, indices[:, 0]] - observed_delta[:, indices[:, 1]]) / 0.1
    tie = (correction_gap.square() * ~active).sum(1) / (~active).sum(1).clamp_min(1)
    keep = (delta / 0.1).square().mean(1)
    weights = state_weights(task_ids, device=delta.device, dtype=delta.dtype)
    parts = {"ranking": (ranking * weights).sum(), "tie": (tie * weights).sum(),
             "keep": (keep * weights).sum()}
    return parts["ranking"] + 0.1 * parts["tie"] + 0.05 * parts["keep"], parts
