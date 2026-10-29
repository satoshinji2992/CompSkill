"""Compile a simple graph-backed anchor, then fit only its conditional residual.

Graph structure preserves execution provenance; uniform aggregation does not
claim higher-order inference. Model selection must compare radius=0 on held-out
training groups. This module cannot read files, call an executor, or score tests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import torch

from skill_experience.retrieval.reliability_key_compiler import fit_frozen_bias_keyopt
from skill_experience.retrieval.skill_evidence_graph import SkillEvidenceGraph


def memory_fingerprint(memory):
    keys, bias = memory.compile_global(allowed_task_ids=[])
    state = dict(snapshot=memory.snapshot(), executor=memory.executor_id,
                 environment=memory.environment_id, refs=memory.refs,
                 keys_shape=list(keys.shape), keys_dtype=str(keys.dtype),
                 base_keys_sha256=hashlib.sha256(keys.cpu().numpy().tobytes()).hexdigest(),
                 base_bias_sha256=hashlib.sha256(bias.cpu().numpy().tobytes()).hexdigest())
    value = json.dumps(state, default=asdict, sort_keys=True,
                       separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(value.encode()).hexdigest()


def validate_memory_version(memory, receipt):
    if memory_fingerprint(memory) != receipt["memory_sha256"]:
        raise ValueError("Evidence changed; recompile the anchor and refit its residual")


def fit_graph_anchored_keyopt(memory, *, allowed_train_task_ids, excluded_source_groups=(),
                            radius=0., steps=500, learning_rate=.03, keep=.01,
                            temperature=.1):
    """Use the same explicit evidence view for anchor and conditional supervision.

    The anchor is the existing uniform task-gradient key plus the memory's fixed
    reliability bias. Only singleton observed arms supply per-skill supervision;
    missing alternatives stay NaN, and bundle outcomes are not attributed to
    their members. All-tie events may affect centering but cannot create pair
    credit. Returned keys exactly fold the residual without row normalization.
    """
    if isinstance(allowed_train_task_ids, (str, bytes)) or isinstance(
            excluded_source_groups, (str, bytes)):
        raise ValueError("Explicit collections of task/group IDs required")
    allowed, excluded = frozenset(allowed_train_task_ids), frozenset(excluded_source_groups)
    before = memory_fingerprint(memory)
    snapshot = memory.snapshot()
    anchor_keys, anchor_bias = memory.compile_global(
        allowed_task_ids=allowed, excluded_source_groups=excluded)
    graph = SkillEvidenceGraph.from_json(snapshot["graph_json"])
    events = graph.view(allowed_train_task_ids=allowed, excluded_source_groups=excluded,
                       executor_id=memory.executor_id, environment_id=memory.environment_id,
                       skill_versions=memory.refs)
    slates = {s.evidence_id: s for s in snapshot["slates"]}
    columns = {ref: i for i, ref in enumerate(memory.refs)}
    queries, labels, masks, event_ids, traces = [], [], [], [], []
    for event in events:
        slate = slates[event.evidence_id]
        arms = [a for a in event.arms if len(a.skills) == 1
                and a.skills[0] in slate.candidates and a.skills[0] in columns]
        if not arms:
            continue
        y = anchor_keys.new_full((len(columns),), float("nan"))
        seen = torch.zeros(len(columns), dtype=torch.bool, device=anchor_keys.device)
        for arm in arms:
            j = columns[arm.skills[0]]
            if bool(seen[j]):
                raise ValueError("Duplicate per-event skill outcome")
            y[j], seen[j] = arm.outcome, True
            traces.append(arm.trace_id)
        queries.append(anchor_keys.new_tensor(slate.query))
        labels.append(y)
        masks.append(seen)
        event_ids.append(event.evidence_id)
    q = torch.stack(queries) if queries else anchor_keys.new_empty((0, anchor_keys.shape[1]))
    y = torch.stack(labels) if labels else anchor_keys.new_empty((0, len(columns)))
    observed = torch.stack(masks) if masks else torch.empty(
        (0, len(columns)), dtype=torch.bool, device=anchor_keys.device)
    if memory_fingerprint(memory) != before:
        raise ValueError("Evidence changed while assembling its anchor")
    index, fitting = fit_frozen_bias_keyopt(anchor_keys, q, y, observed, anchor_bias,
        steps=steps, learning_rate=learning_rate, radius=radius, keep=keep, temperature=temperature)
    receipt = dict(memory_sha256=before, allowed_train_task_ids=sorted(allowed),
        excluded_source_groups=sorted(excluded), evidence_ids=event_ids, trace_ids=traces,
        executor_id=memory.executor_id, environment_id=memory.environment_id,
        skill_refs=[asdict(r) for r in memory.refs], fitting=fitting,
        observed_arms=int(observed.sum()), anchor="uniform_task_updates_plus_fixed_bias",
        anchor_keys_sha256=hashlib.sha256(anchor_keys.cpu().numpy().tobytes()).hexdigest(),
        anchor_bias_sha256=hashlib.sha256(anchor_bias.cpu().numpy().tobytes()).hexdigest(),
        requested_radius=radius, no_update=bool(index.delta.count_nonzero() == 0),
        scope="Evidence-backed compilation; nonzero residual has no guaranteed success gain")
    validate_memory_version(memory, receipt)
    return index, receipt
