"""Task-slate KeyOpt deltas, stored and read as indivisible evidence hyperedges.

One task produces one joint rank-one update from observed alternative executions.
Online conditional reading is deliberately NOT a static single-key index. No I/O,
model calls, outcome inference, or automatic collection occurs in this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from skill_experience.retrieval.skill_evidence_graph import (
    EpisodeEvidence,
    SkillEvidenceGraph,
    SkillRef,
    _identifier,
    _sha256,
)


@dataclass(frozen=True)
class RetrievalSlate:
    evidence_id: str
    task_id: str
    source_group: str
    query_sha256: str
    context_sha256: str
    executor_id: str
    environment_id: str
    seed: int
    candidates: tuple[SkillRef, ...]
    scores: tuple[float, ...]
    query: tuple[float, ...]
    memory_before_sha256: str
    candidate_source: str = "retrieved"
    candidate_source_sha256: str | None = None


@dataclass(frozen=True)
class TaskKeyDelta:
    evidence_id: str
    candidates: tuple[SkillRef, ...]
    coefficients: tuple[float, ...]
    observed_count: int
    reason: str


def task_key_gradient(scores: Tensor, outcomes: Tensor, observed: Tensor, *,
                      learning_rate=.05, temperature=.1, radius=.05):
    """One exact expected-utility ascent direction, then joint trust-region scaling.

    For unit query x, delta_k[s] = coefficient[s] * x. Observed alternatives
    share p=softmax(score/temperature) and U=sum(p*y). The ascent coefficient is
    lr/temperature * p[s]*(y[s]-U). Missing/tied/single-arm feedback cannot create
    a relative update. This is an observed-slate objective, not a propensity-
    corrected estimate over unexecuted alternatives.
    """
    if (scores.ndim != 1 or outcomes.shape != scores.shape or observed.shape != scores.shape
            or observed.dtype != torch.bool or scores.dtype not in (torch.float32, torch.float64)
            or outcomes.dtype != scores.dtype or scores.device != outcomes.device
            or observed.device != scores.device or not torch.isfinite(scores).all()
            or not torch.isfinite(outcomes[observed]).all()
            or not ((outcomes[observed] == 0) | (outcomes[observed] == 1)).all()):
        raise ValueError('Invalid observed task-slate feedback')
    for value in (learning_rate, temperature, radius):
        if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('Positive finite update configuration required')
    result = torch.zeros_like(scores)
    y = outcomes[observed]
    if len(y) < 2 or bool((y == y[0]).all()):
        return result
    p = (scores.detach()[observed] / temperature).softmax(0)
    alpha = learning_rate / temperature * p * (y - (p*y).sum())
    # Remove roundoff only: the analytical full-slate gradient sums to zero.
    alpha = alpha - alpha.mean()
    alpha = alpha * (radius / alpha.norm().clamp_min(radius))
    result[observed] = alpha
    return result.detach()


class TaskKeyOptMemory:
    """Version-bound past-task hypergraph plus compact per-event key corrections.

    Graph membership preserves an entire alternative-comparison event. A single
    query-conditioned weight reads each event's complete coefficient vector;
    selecting members independently would change the learned relative update.
    A lossless event-node graph can implement the same representation.
    """

    def __init__(self, keys, skill_refs, *, executor_id, environment_id, bias=None,
                 learning_rate=.05, temperature=.1, radius=.05, memory_temperature=.1,
                 memory_top_m=16, similarity_floor=.3):
        refs = tuple(skill_refs)
        if (keys.ndim != 2 or not all(keys.shape) or keys.dtype != torch.float32
                or keys.requires_grad or not torch.isfinite(keys).all()
                or len(refs) != len(keys) or any(not isinstance(r, SkillRef) for r in refs)
                or len({r.skill_id for r in refs}) != len(refs)):
            raise ValueError('Detached float32 keys and unique versioned skill references required')
        _identifier(executor_id, 'executor_id')
        _identifier(environment_id, 'environment_id')
        for val in (learning_rate, temperature, radius, memory_temperature):
            if not isinstance(val, (float, int)) or isinstance(val, bool) or not math.isfinite(val) or val <= 0:
                raise ValueError('Positive finite configuration required')
        if type(memory_top_m) is not int or memory_top_m < 1 or not -1 <= similarity_floor < 1:
            raise ValueError('Invalid memory readout settings')
        self._keys = keys.detach().clone()
        self._bias = keys.new_zeros(len(keys)) if bias is None else bias.detach().clone()
        if (self._bias.shape != (len(keys),) or self._bias.dtype != keys.dtype
                or self._bias.device != keys.device or not torch.isfinite(self._bias).all()):
            raise ValueError('Invalid fixed reliability bias')
        self.refs, self.executor_id, self.environment_id = refs, executor_id, environment_id
        self._lookup = {r: i for i, r in enumerate(refs)}
        self._graph = SkillEvidenceGraph()
        self._slates: dict[str, RetrievalSlate] = {}
        self._updates: dict[str, TaskKeyDelta] = {}
        self.learning_rate, self.temperature, self.radius = learning_rate, temperature, radius
        self.memory_temperature, self.memory_top_m = memory_temperature, memory_top_m
        self.similarity_floor = similarity_floor

    def _query(self, query):
        if (query.shape != (self._keys.shape[1],) or query.dtype != self._keys.dtype
                or query.device != self._keys.device or query.requires_grad
                or not torch.isfinite(query).all()
                or not torch.isclose(query.norm(), query.new_tensor(1.), atol=1e-5, rtol=1e-5)):
            raise ValueError('Detached unit query in the frozen key space required')
        return query

    def _events(self, allowed_task_ids, excluded_source_groups):
        allowed = ({s.task_id for s in self._slates.values()} if allowed_task_ids is None
                   else allowed_task_ids)
        events = self._graph.view(allowed_train_task_ids=allowed,
            excluded_source_groups=excluded_source_groups, executor_id=self.executor_id,
            environment_id=self.environment_id, skill_versions=self.refs)
        return [e for e in events if any(self._updates[e.evidence_id].coefficients)]

    def _read(self, query, mode, allowed_task_ids, excluded_source_groups):
        if mode not in ('conditional', 'global', 'none'):
            raise ValueError('Unknown memory mode')
        if mode == 'none':
            return [], query.new_empty(0), query.new_empty(0)
        events = self._events(allowed_task_ids, excluded_source_groups)
        if not events:
            return [], query.new_empty(0), query.new_empty(0)
        event_queries = query.new_tensor([self._slates[e.evidence_id].query for e in events])
        similarity = event_queries @ query
        if mode == 'global':
            return events, torch.ones_like(similarity)/len(events), similarity
        # Public-query-only matching. Below-threshold evidence backs off exactly.
        valid = (similarity > self.similarity_floor).nonzero(as_tuple=True)[0]
        order = valid[torch.argsort(similarity[valid], descending=True, stable=True)[:self.memory_top_m]]
        if not len(order):
            return [], query.new_empty(0), query.new_empty(0)
        selected_similarity = similarity[order]
        # Null token prevents forced updates for weakly matched histories.
        logits = torch.cat(((selected_similarity-self.similarity_floor)/self.memory_temperature,
                            query.new_zeros(1)))
        weights = logits.softmax(0)[:-1]
        return [events[i] for i in order.tolist()], weights, selected_similarity

    @torch.no_grad()
    def score(self, query, *, mode='conditional', allowed_task_ids=None,
              excluded_source_groups=(), return_details=False):
        query = self._query(query)
        scores = self._keys @ query + self._bias
        events, weights, similarities = self._read(query, mode, allowed_task_ids, excluded_source_groups)
        correction = torch.zeros_like(scores)
        for event, weight, similarity in zip(events, weights, similarities, strict=True):
            update = self._updates[event.evidence_id]
            ids = torch.tensor([self._lookup[r] for r in update.candidates], device=query.device)
            correction.index_add_(0, ids, weight * similarity * query.new_tensor(update.coefficients))
        result = scores + correction
        details = dict(evidence_ids=[e.evidence_id for e in events], weights=weights,
                       correction=correction, mode=mode)
        return (result, details) if return_details else result

    @torch.no_grad()
    def retrieve(self, query, *, evidence_id, task_id, source_group, query_sha256,
                 context_sha256, seed, top_k=8, mode='conditional', allowed_task_ids=None,
                 excluded_source_groups=(), candidate_refs=None, selection_sha256=None):
        """Register a pre-feedback candidate slate; caller runs existing executor."""
        for value, name in [(evidence_id, 'evidence_id'), (task_id, 'task_id'), (source_group, 'source_group')]:
            _identifier(value, name)
        _sha256(query_sha256, 'query_sha256')
        _sha256(context_sha256, 'context_sha256')
        if type(seed) is not int or type(top_k) is not int or top_k < 1:
            raise ValueError('Explicit seed and positive TopK required')
        if evidence_id in self._slates:
            raise ValueError('Duplicate retrieval event')
        scores = self.score(query, mode=mode, allowed_task_ids=allowed_task_ids,
                            excluded_source_groups=excluded_source_groups)
        if candidate_refs is None:
            if selection_sha256 is not None:
                raise ValueError('Historical selection hash without historical candidates')
            ids = torch.argsort(scores, descending=True, stable=True)[:top_k]
            candidate_source = 'retrieved'
        else:
            _sha256(selection_sha256, 'selection_sha256')
            candidates = tuple(candidate_refs)
            if (not candidates or len(candidates) > top_k or len(set(candidates)) != len(candidates)
                    or any(r not in self._lookup for r in candidates)):
                raise ValueError('Invalid frozen historical candidate slate')
            ids = torch.tensor([self._lookup[r] for r in candidates], device=query.device)
            candidate_source = 'historical_receipt'
        slate = RetrievalSlate(evidence_id, task_id, source_group, query_sha256, context_sha256,
            self.executor_id, self.environment_id, seed, tuple(self.refs[i] for i in ids.tolist()),
            tuple(scores[ids].tolist()), tuple(query.tolist()), self._graph.snapshot_sha256, candidate_source, selection_sha256)
        self._slates[evidence_id] = slate
        return slate

    @torch.no_grad()
    def record(self, slate, episode):
        """Commit exactly once; only actually executed singleton arms update keys."""
        if not isinstance(slate, RetrievalSlate) or self._slates.get(slate.evidence_id) != slate:
            raise ValueError('Unknown or altered pre-feedback slate')
        if not isinstance(episode, EpisodeEvidence) or episode.split != 'train':
            raise ValueError('Only explicitly authorized TRAIN execution feedback is admitted')
        for name in ('evidence_id', 'task_id', 'source_group', 'query_sha256', 'context_sha256',
                     'executor_id', 'environment_id', 'seed'):
            if getattr(slate, name) != getattr(episode, name):
                raise ValueError('Execution does not match the retrieval context: '+name)
        if episode.evidence_id in self._updates:
            raise ValueError('Task feedback already committed')
        scores = self._keys.new_tensor(slate.scores)
        y, seen = scores.new_full(scores.shape, float('nan')), torch.zeros_like(scores, dtype=torch.bool)
        positions = {r: i for i, r in enumerate(slate.candidates)}
        for arm in episode.arms:
            if any(r not in positions for r in arm.skills):
                raise ValueError('Observed skill version is outside the frozen candidate slate')
            if len(arm.skills) != 1:
                continue  # A bundle/no-skill outcome is not individual skill credit.
            i = positions[arm.skills[0]]
            if seen[i]:
                raise ValueError('Repeated singleton arm cannot be collapsed into one result')
            y[i], seen[i] = arm.outcome, True
        alpha = task_key_gradient(scores, y, seen, learning_rate=self.learning_rate,
                                  temperature=self.temperature, radius=self.radius)
        reason = 'joint_relative_update' if bool(alpha.any()) else 'no_observed_relative_advantage'
        update = TaskKeyDelta(slate.evidence_id, slate.candidates, tuple(alpha.tolist()), int(seen.sum()), reason)
        self._graph.append(episode)  # Checks duplicate physical traces before any mutation below.
        self._updates[episode.evidence_id] = update
        return update

    def deactivate(self, evidence_id, reason):
        self._graph.deactivate(evidence_id, reason)

    def reactivate(self, evidence_id, reason):
        self._graph.reactivate(evidence_id, reason)

    @torch.no_grad()
    def compile_global(self, *, allowed_task_ids=None, excluded_source_groups=()):
        """Uniform event averaging as a static-KeyOpt ablation; no renormalization."""
        keys = self._keys.clone()
        events = self._events(allowed_task_ids, excluded_source_groups)
        for event in events:
            slate, update = self._slates[event.evidence_id], self._updates[event.evidence_id]
            ids = torch.tensor([self._lookup[r] for r in update.candidates], device=keys.device)
            delta = keys.new_tensor(update.coefficients)[:, None]*keys.new_tensor(slate.query)[None]
            keys.index_add_(0, ids, delta/len(events))
        return keys, self._bias.clone()

    def snapshot(self):
        """Immutable provenance objects plus canonical graph; no file writes."""
        return dict(graph_json=self._graph.to_json(), slates=tuple(self._slates.values()),
                    updates=tuple(self._updates.values()))
