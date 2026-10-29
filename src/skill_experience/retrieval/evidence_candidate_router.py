"""Bounded candidate expansion from versioned task-KeyOpt evidence.

Reuses TaskKeyOptMemory's event gate and signed updates. This is not GoSkills,
and alternative arms are never interpreted as a jointly useful skill bundle.
The caller supplies frozen semantic scores (or a reranker's scores); no I/O,
model loading, outcome collection, or fitting takes place here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from skill_experience.retrieval.graph_anchored_keyopt import memory_fingerprint
from skill_experience.retrieval.skill_evidence_graph import SkillRef
from skill_experience.retrieval.task_key_memory import TaskKeyOptMemory


@dataclass(frozen=True)
class CandidateDecision:
    semantic_candidates: tuple[SkillRef, ...]
    candidates: tuple[SkillRef, ...]
    ranked: tuple[SkillRef, ...]
    added: tuple[SkillRef, ...]
    evidence_ids: tuple[str, ...]
    event_weights: tuple[float, ...]
    scores: tuple[float, ...]
    corrections: tuple[float, ...]
    memory_sha256: str
    use_keyopt: bool


class EvidenceCandidateRouter:
    def __init__(self, memory: TaskKeyOptMemory, *, candidate_budget=10,
                 protected_semantic=6, max_expansion=4):
        if not isinstance(memory, TaskKeyOptMemory):
            raise TypeError("Existing TaskKeyOptMemory required")
        if any(type(x) is not int for x in
               (candidate_budget, protected_semantic, max_expansion)):
            raise ValueError("Integer budgets required")
        if not (0 < candidate_budget <= len(memory.refs)
                and 1 <= protected_semantic <= candidate_budget
                and 0 <= max_expansion <= candidate_budget - protected_semantic):
            raise ValueError("Invalid protected candidate budget")
        self.memory = memory
        self.candidate_budget = candidate_budget
        self.protected_semantic = protected_semantic
        self.max_expansion = max_expansion

    @torch.no_grad()
    def route(self, context_query, semantic_scores, *, current_task_id,
              current_source_group, allowed_history_task_ids,
              excluded_source_groups=(), use_keyopt=True, expand=True,
              fixed_bias=None):
        """Use only an explicit past-task allowlist; always exclude this group.

        Both bias-only and KeyOpt arms share candidate construction. Score order
        matches ``ranked``; corrections describe the actually applied residual.
        Missing/reranker scores must not be fabricated: all bank rows are needed
        to assemble candidates; a later reranker may score only this fixed pool.
        """
        if not isinstance(current_task_id, str) or not current_task_id:
            raise ValueError("Current task identity required")
        if not isinstance(current_source_group, str) or not current_source_group:
            raise ValueError("Current source group required")
        if isinstance(allowed_history_task_ids, (str, bytes)) or allowed_history_task_ids is None:
            raise ValueError("Explicit history task allowlist required")
        if isinstance(excluded_source_groups, (str, bytes)):
            raise ValueError("Explicit group exclusion list required")
        allowed = frozenset(allowed_history_task_ids) - {current_task_id}
        excluded = frozenset(excluded_source_groups) | {current_source_group}
        n = len(self.memory.refs)
        if (semantic_scores.shape != (n,) or semantic_scores.requires_grad
                or not torch.isfinite(semantic_scores).all()
                or semantic_scores.dtype != context_query.dtype
                or semantic_scores.device != context_query.device):
            raise ValueError("Finite detached, bank-aligned semantic scores required")
        bias = torch.zeros_like(semantic_scores) if fixed_bias is None else fixed_bias
        if (bias.shape != (n,) or bias.requires_grad or not torch.isfinite(bias).all()
                or bias.dtype != semantic_scores.dtype or bias.device != semantic_scores.device):
            raise ValueError("Invalid fixed bias")
        before = memory_fingerprint(self.memory)
        _, details = self.memory.score(context_query, mode="conditional",
            allowed_task_ids=allowed, excluded_source_groups=excluded, return_details=True)
        correction = details["correction"]
        semantic_order = torch.argsort(semantic_scores, descending=True, stable=True).tolist()
        original = semantic_order[:self.candidate_budget]
        selected = original[:self.protected_semantic]
        extra = []
        if expand:
            # Only a positively supported, actually observed member can expand
            # the pool. Unexecuted slate members have exact zero correction.
            for i in torch.argsort(correction, descending=True, stable=True).tolist():
                if len(extra) >= self.max_expansion:
                    break
                if i not in original and float(correction[i]) > 0:
                    extra.append(i)
        selected += extra
        for i in semantic_order:
            if len(selected) == self.candidate_budget:
                break
            if i not in selected:
                selected.append(i)
        applied = correction if use_keyopt else torch.zeros_like(correction)
        scores = semantic_scores + bias + applied
        ranked = sorted(selected, key=lambda i: (-float(scores[i]), i))
        if before != memory_fingerprint(self.memory):
            raise ValueError("Evidence changed during routing")
        refs = self.memory.refs
        return CandidateDecision(
            tuple(refs[i] for i in original), tuple(refs[i] for i in selected),
            tuple(refs[i] for i in ranked), tuple(refs[i] for i in extra),
            tuple(details["evidence_ids"]), tuple(details["weights"].tolist()),
            tuple(float(scores[i]) for i in ranked),
            tuple(float(applied[i]) for i in ranked), before, bool(use_keyopt))
