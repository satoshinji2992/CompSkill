"""Append-only evidence hypergraph for versioned skill execution observations.

An episode binds a task/context/seed/environment to mutually exclusive tested
arms. Alternatives imply neither composition nor conflict. An arm with multiple
skills records observed co-use only: its reward belongs to that whole bundle.
This module provides provenance/filtering, not an online scorer or causal claims.
It uses only caller-supplied observations and performs no file/service/model I/O.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from numbers import Integral


def _identifier(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


def _sha256(value, field):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256 hex digest")


@dataclass(frozen=True, slots=True)
class SkillRef:
    skill_id: str
    document_sha256: str
    lesson_sha256: str | None = None

    def __post_init__(self):
        _identifier(self.skill_id, "skill_id")
        _sha256(self.document_sha256, "document_sha256")
        if self.lesson_sha256 is not None:
            _sha256(self.lesson_sha256, "lesson_sha256")


@dataclass(frozen=True, slots=True)
class ObservedArm:
    """One actual execution; empty skills denotes an observed no-skill control.

    Missing outcomes are rejected. Callers must omit unobserved arms instead of
    representing missing feedback as failure. No constituent inherits a bundle's
    terminal outcome; a bundle cannot be trimmed by a version filter.
    """

    skills: tuple[SkillRef, ...]
    outcome: int
    trace_id: str

    def __post_init__(self):
        try:
            skills = tuple(self.skills)
        except TypeError as error:
            raise ValueError("skills must contain immutable SkillRef objects") from error
        if not all(isinstance(skill, SkillRef) for skill in skills):
            raise ValueError("skills must contain immutable SkillRef objects")
        if len(set(skills)) != len(skills):
            raise ValueError("An arm cannot repeat the same skill version")
        if not isinstance(self.outcome, Integral) or self.outcome not in (0, 1):
            raise ValueError("outcome must be observed binary 0/1; missing is not failure")
        _identifier(self.trace_id, "trace_id")
        object.__setattr__(self, "skills", skills)
        object.__setattr__(self, "outcome", int(self.outcome))


@dataclass(frozen=True, slots=True)
class EpisodeEvidence:
    """Matched observed alternatives sharing all explicit comparability fields."""

    evidence_id: str
    task_id: str
    source_group: str
    query_sha256: str
    split: str
    executor_id: str
    environment_id: str
    seed: int
    context_sha256: str
    arms: tuple[ObservedArm, ...]

    def __post_init__(self):
        for name in ("evidence_id", "task_id", "source_group", "split",
                     "executor_id", "environment_id"):
            _identifier(getattr(self, name), name)
        _sha256(self.query_sha256, "query_sha256")
        _sha256(self.context_sha256, "context_sha256")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("seed must be an explicit integer")
        try:
            arms = tuple(self.arms)
        except TypeError as error:
            raise ValueError("arms must contain observed executions") from error
        if not arms or not all(isinstance(arm, ObservedArm) for arm in arms):
            raise ValueError("arms must contain at least one ObservedArm")
        if len({arm.trace_id for arm in arms}) != len(arms):
            raise ValueError("Duplicate trace ID within episode")
        object.__setattr__(self, "arms", arms)
        object.__setattr__(self, "seed", int(self.seed))


@dataclass(frozen=True, slots=True)
class EvidenceStatus:
    evidence_id: str
    active: bool
    reason: str


class SkillEvidenceGraph:
    """Mutable append-only journal until freeze(), with immutable observations.

    Deactivation/reactivation appends a reasoned status event. It never deletes
    evidence or releases its trace identity. freeze() permanently freezes this
    instance and returns it; canonical snapshot roundtrips preserve that state.
    """

    def __init__(self):
        self._episodes: dict[str, EpisodeEvidence] = {}
        self._trace_ids: set[str] = set()
        self._active: dict[str, bool] = {}
        self._journal: list[EpisodeEvidence | EvidenceStatus] = []
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def history(self) -> tuple[EpisodeEvidence | EvidenceStatus, ...]:
        return tuple(self._journal)

    def __len__(self) -> int:
        return len(self._episodes)

    def _require_mutable(self):
        if self._frozen:
            raise RuntimeError("Frozen evidence graph cannot be mutated")

    def append(self, episode: EpisodeEvidence) -> None:
        self._require_mutable()
        if not isinstance(episode, EpisodeEvidence):
            raise ValueError("append requires an EpisodeEvidence")
        if episode.evidence_id in self._episodes:
            raise ValueError("Duplicate evidence ID")
        traces = {arm.trace_id for arm in episode.arms}
        if traces & self._trace_ids:
            raise ValueError("Duplicate trace ID; execution evidence cannot be counted twice")
        self._episodes[episode.evidence_id] = episode
        self._trace_ids.update(traces)
        self._active[episode.evidence_id] = True
        self._journal.append(episode)

    def _set_active(self, evidence_id: str, active: bool, reason: str):
        self._require_mutable()
        _identifier(reason, "reason")
        if evidence_id not in self._episodes:
            raise ValueError("Unknown evidence ID")
        if self._active[evidence_id] == active:
            raise ValueError("Evidence already has the requested status")
        self._active[evidence_id] = active
        self._journal.append(EvidenceStatus(evidence_id, active, reason))

    def deactivate(self, evidence_id: str, reason: str) -> None:
        self._set_active(evidence_id, False, reason)

    def reactivate(self, evidence_id: str, reason: str) -> None:
        self._set_active(evidence_id, True, reason)

    def freeze(self) -> SkillEvidenceGraph:
        self._frozen = True
        return self

    def view(self, *, allowed_train_task_ids, excluded_source_groups,
             executor_id: str, environment_id: str,
             skill_versions, seed: int | None = None) -> tuple[EpisodeEvidence, ...]:
        """Return active TRAIN-only evidence under an explicit allowlist.

        Source-group exclusion drops the whole episode. Each retained arm must
        match every document/lesson version exactly; None means *no lesson*, not
        any lesson. Unobserved/mismatched arms are omitted, never zero-filled.
        Filtering leaves the stored episode and the journal unchanged. Empty
        no-skill controls can remain, but are never exported as individual skills.
        """
        _identifier(executor_id, "executor_id")
        _identifier(environment_id, "environment_id")
        if (isinstance(allowed_train_task_ids, (str, bytes))
                or isinstance(excluded_source_groups, (str, bytes))):
            raise ValueError("Task/group filters must be collections of explicit IDs")
        try:
            allowed = frozenset(allowed_train_task_ids)
            excluded = frozenset(excluded_source_groups)
            versions = frozenset(skill_versions)
        except TypeError as error:
            raise ValueError("Filters must be collections of IDs and SkillRef objects") from error
        for value in allowed | excluded:
            _identifier(value, "task/group filter")
        if not all(isinstance(ref, SkillRef) for ref in versions):
            raise ValueError("skill_versions must contain exact SkillRef objects")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
            raise ValueError("seed filter must be an integer")
        result = []
        for episode in self._episodes.values():
            if (not self._active[episode.evidence_id] or episode.split != "train"
                    or episode.task_id not in allowed or episode.source_group in excluded
                    or episode.executor_id != executor_id or episode.environment_id != environment_id
                    or (seed is not None and episode.seed != seed)):
                continue
            arms = tuple(arm for arm in episode.arms if all(ref in versions for ref in arm.skills))
            if arms:
                result.append(replace(episode, arms=arms))
        return tuple(result)

    def single_arm_observations(self, **view_filters) -> tuple[dict, ...]:
        """Export traceable single-skill observations, with no bundle attribution.

        Every returned dictionary is a fresh copy. This deliberately performs no
        aggregation or missing-label filling; callers can count only these actual
        observations, with the full provenance available for their own ledger.
        """
        rows = []
        for episode in self.view(**view_filters):
            context = {name: getattr(episode, name) for name in
                       ("evidence_id", "task_id", "source_group", "query_sha256", "split",
                        "executor_id", "environment_id", "seed", "context_sha256")}
            for arm in episode.arms:
                if len(arm.skills) == 1:
                    rows.append({**context, **asdict(arm.skills[0]),
                                 "outcome": arm.outcome, "trace_id": arm.trace_id})
        return tuple(rows)

    def to_json(self) -> str:
        events = [{"kind": "episode" if isinstance(event, EpisodeEvidence) else "status",
                   "value": asdict(event)} for event in self._journal]
        return json.dumps({"schema_version": 1, "frozen": self._frozen, "events": events},
                          sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

    @property
    def snapshot_sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, value: str) -> SkillEvidenceGraph:
        """Validate/replay a canonical or ordinary JSON snapshot without any I/O."""
        def unique_object(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON object key")
                result[key] = item
            return result

        try:
            snapshot = json.loads(value, object_pairs_hook=unique_object)
            if (not isinstance(snapshot, dict)
                    or set(snapshot) != {"schema_version", "frozen", "events"}
                    or type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != 1
                    or not isinstance(snapshot["frozen"], bool)
                    or not isinstance(snapshot["events"], list)):
                raise ValueError("Invalid evidence graph snapshot schema")
            graph = cls()
            for event in snapshot["events"]:
                if not isinstance(event, dict) or set(event) != {"kind", "value"}:
                    raise ValueError("Invalid journal event")
                fields = event["value"]
                if event["kind"] == "episode":
                    arms = tuple(ObservedArm(
                        **{**arm, "skills": tuple(SkillRef(**ref) for ref in arm["skills"])})
                        for arm in fields["arms"])
                    graph.append(EpisodeEvidence(**{**fields, "arms": arms}))
                elif event["kind"] == "status":
                    status = EvidenceStatus(**fields)
                    if not isinstance(status.active, bool):
                        raise ValueError("Status active must be boolean")
                    graph._set_active(status.evidence_id, status.active, status.reason)
                else:
                    raise ValueError("Unknown journal event kind")
            if snapshot["frozen"]:
                graph.freeze()
            return graph
        except (TypeError, KeyError) as error:
            raise ValueError("Malformed evidence graph snapshot") from error
