"""Versioned bundle feedback; terminal group rewards never update member keys.

This is a small, CPU-only extension of the existing observed-slate KeyOpt rule.
Exact hyperedges and lossless event nodes are equivalent views. Pair projection
is an explicitly lossy ablation, not evidence that each pair was executed.
"""
from __future__ import annotations

import hashlib
import copy
import itertools
import json
import math
from dataclasses import asdict, dataclass

import torch

from skill_experience.retrieval.skill_evidence_graph import SkillRef
from skill_experience.retrieval.task_key_memory import task_key_gradient


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class SkillGroup:
    """Ordered documents actually presented, including their assigned roles."""
    members: tuple[SkillRef, ...]
    roles: tuple[str, ...]
    presentation_version: str

    def __post_init__(self):
        object.__setattr__(self, "members", tuple(self.members))
        object.__setattr__(self, "roles", tuple(self.roles))
        if (not 1 <= len(self.members) <= 4 or len(self.members) != len(self.roles)
                or len({r.skill_id for r in self.members}) != len(self.members)
                or not all(isinstance(r, SkillRef) for r in self.members)
                or not all(isinstance(r, str) and r.strip() for r in self.roles)
                or not self.presentation_version):
            raise ValueError("Unique versioned members, roles and presentation version required")

    @property
    def identity(self):
        return canonical_hash(asdict(self))

    @property
    def pairs(self):
        # Relative order and roles survive projection. Singleton credit is never
        # manufactured from a larger group's terminal result.
        nodes = [(asdict(m), r) for m, r in zip(self.members, self.roles, strict=True)]
        return frozenset(canonical_hash([self.presentation_version, a, b])
                         for a, b in itertools.combinations(nodes, 2))


@dataclass(frozen=True)
class GroupSlate:
    event_id: str
    task_id: str
    source_group: str
    executor_id: str
    environment_id: str
    repeat: int
    step: int
    query: tuple[float, ...]
    groups: tuple[SkillGroup, ...]
    scores: tuple[float, ...]
    memory_sha256: str


class GroupEvidenceMemory:
    def __init__(self, *, dimension=512, radius=.05, temperature=.1,
                 learning_rate=.05, similarity_floor=.3):
        if type(dimension) is not int or dimension < 1:
            raise ValueError("Positive dimension required")
        if not all(math.isfinite(v) and v > 0 for v in (radius, temperature, learning_rate)):
            raise ValueError("Positive finite update settings required")
        if not -1 <= similarity_floor < 1:
            raise ValueError("Invalid similarity floor")
        self.dimension, self.radius, self.temperature = dimension, radius, temperature
        self.learning_rate, self.similarity_floor = learning_rate, similarity_floor
        self._slates, self._events, self._traces = {}, {}, set()

    @property
    def fingerprint(self):
        return canonical_hash(self.export())

    def _query(self, query):
        q = torch.as_tensor(query, dtype=torch.float32, device="cpu").detach()
        if q.shape != (self.dimension,) or not torch.isfinite(q).all() or not torch.isclose(
                q.norm(), torch.tensor(1.), atol=1e-5, rtol=1e-5):
            raise ValueError("Finite unit query in the frozen key space required")
        return q

    def register(self, *, event_id, task_id, source_group, executor_id, environment_id,
                 repeat, step, query, groups, scores):
        if not all(isinstance(v, str) and v for v in
                   (event_id, task_id, source_group, executor_id, environment_id)):
            raise ValueError("Explicit event/task/source/executor/environment IDs required")
        if event_id in self._slates or type(step) is not int or step < 0 or type(repeat) is not int:
            raise ValueError("Duplicate event or invalid causal position")
        groups, scores = tuple(groups), tuple(float(s) for s in scores)
        if (not groups or len(groups) != len(scores)
                or not all(isinstance(g, SkillGroup) for g in groups)
                or len({g.identity for g in groups}) != len(groups)
                or not all(math.isfinite(s) for s in scores)):
            raise ValueError("Unique group alternatives and finite pre-feedback scores required")
        slate = GroupSlate(event_id, task_id, source_group, executor_id, environment_id,
                           repeat, step, tuple(self._query(query).tolist()), groups,
                           scores, self.fingerprint)
        self._slates[event_id] = slate
        return slate

    def commit(self, slate, *, observed, available_step, split="train"):
        """observed maps group identity to {outcome, trace_id}; missing stays unknown."""
        if (self._slates.get(slate.event_id) != slate or slate.event_id in self._events
                or split not in ("train", "online") or type(available_step) is not int
                or available_step <= slate.step):
            raise ValueError("Feedback requires its original slate and a later availability step")
        ids = [g.identity for g in slate.groups]
        if not observed or not set(observed) <= set(ids):
            raise ValueError("Only actually observed alternatives from this slate are admitted")
        traces = [v["trace_id"] for v in observed.values()]
        if (not all(isinstance(t, str) and t for t in traces) or len(set(traces)) != len(traces)
                or self._traces.intersection(traces)):
            raise ValueError("Duplicate or missing physical execution trace")
        if any(type(v["outcome"]) is not int or v["outcome"] not in (0, 1)
               for v in observed.values()):
            raise ValueError("Observed binary outcomes required; unknown is not failure")
        mask = torch.tensor([i in observed for i in ids], dtype=torch.bool)
        outcomes = torch.tensor([float(observed[i]["outcome"]) if i in observed else float("nan")
                                 for i in ids])
        delta = task_key_gradient(torch.tensor(slate.scores), outcomes, mask,
            radius=self.radius, temperature=self.temperature, learning_rate=self.learning_rate)
        event = dict(slate=asdict(slate), observed={k: dict(v) for k, v in observed.items()},
                     coefficients=delta.tolist(), available_step=available_step, split=split)
        self._events[slate.event_id] = event
        self._traces.update(traces)
        return copy.deepcopy(event)

    def corrections(self, groups, query, *, allowed_history_task_ids, current_task_id,
                    current_source_group, executor_id, environment_id, repeat, step,
                    view="hypergraph", conditional=True):
        if view not in ("hypergraph", "event_nodes", "pairwise", "none"):
            raise ValueError("Unknown evidence view")
        if (allowed_history_task_ids is None or isinstance(allowed_history_task_ids, (str, bytes))
                or not current_task_id or not current_source_group
                or not executor_id or not environment_id or type(step) is not int
                or type(repeat) is not int):
            raise ValueError("Explicit history allowlist and causal query identity required")
        allowed = set(allowed_history_task_ids) - {current_task_id}
        q = self._query(query)
        result = {}
        for group in groups:
            by_task = {}
            for eid, event in self._events.items():
                slate = self._slates[eid]
                if (slate.task_id not in allowed or slate.source_group == current_source_group
                        or slate.executor_id != executor_id or slate.environment_id != environment_id
                        or event["available_step"] > step
                        or (event["split"] == "online" and slate.repeat != repeat) or view == "none"):
                    continue
                sim = float(torch.dot(q, torch.tensor(slate.query)))
                for old, coefficient in zip(slate.groups, event["coefficients"], strict=True):
                    feedback = event["observed"].get(old.identity)
                    if feedback is None:
                        continue
                    exact = old.identity == group.identity
                    match = float(exact)
                    if view == "pairwise" and not exact and group.pairs:
                        # Explicitly inferred compatibility, not observed pair reward.
                        match = len(group.pairs & old.pairs) / len(group.pairs)
                    if not match:
                        continue
                    gate = max(0., (sim - self.similarity_floor) / (1 - self.similarity_floor))
                    row = (match * (2 * feedback["outcome"] - 1),
                           match * gate * sim * coefficient if conditional else 0., eid)
                    by_task.setdefault(slate.task_id, []).append(row)
            # A task with many repeats/branches must not count as many independent tasks.
            bias, delta, evidence = 0., 0., set()
            for rows in by_task.values():
                bias += sum(r[0] for r in rows) / len(rows)
                delta += sum(r[1] for r in rows) / len(rows)
                evidence.update(r[2] for r in rows)
            denom = len(by_task) + 1  # zero-update prior, bounded shrinkage
            result[group.identity] = dict(bias=self.radius * bias / denom,
                keyopt=delta / denom, support_tasks=len(by_task), evidence_ids=sorted(evidence))
        return result

    def export(self):
        return dict(schema=1, dimension=self.dimension, radius=self.radius,
                    temperature=self.temperature, learning_rate=self.learning_rate,
                    similarity_floor=self.similarity_floor, events=copy.deepcopy(list(self._events.values())))

    @classmethod
    def restore(cls, data):
        if data["schema"] != 1:
            raise ValueError("Unknown group evidence schema")
        memory = cls(**{k: data[k] for k in ("dimension", "radius", "temperature",
                                           "learning_rate", "similarity_floor")})
        for event in data["events"]:
            raw = dict(event["slate"])
            expected_hash = raw.pop("memory_sha256")
            if expected_hash != memory.fingerprint:
                raise ValueError("History snapshot chain mismatch")
            raw["groups"] = tuple(SkillGroup(tuple(SkillRef(**m) for m in g["members"]),
                tuple(g["roles"]), g["presentation_version"]) for g in raw["groups"])
            slate = memory.register(**raw)
            regenerated = memory.commit(slate, observed=event["observed"],
                available_step=event["available_step"], split=event["split"])
            if regenerated["coefficients"] != event["coefficients"]:
                raise ValueError("Changed update coefficients")
        return memory
