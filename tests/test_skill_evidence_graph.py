"""Synthetic evidence only: isolation, attribution, journal rollback and freeze."""

import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from skill_experience.retrieval.skill_evidence_graph import (
    EpisodeEvidence,
    EvidenceStatus,
    ObservedArm,
    SkillEvidenceGraph,
    SkillRef,
)

A = SkillRef("a", "a" * 64)
B = SkillRef("b", "b" * 64, "c" * 64)


def episode(evidence_id="e", **changes):
    return replace(EpisodeEvidence(
        evidence_id=evidence_id, task_id="train-1", source_group="database-1",
        query_sha256="d" * 64, split="train", executor_id="executor-a",
        environment_id="environment-a", seed=0, context_sha256="e" * 64,
        arms=(ObservedArm((A,), 1, f"{evidence_id}/a"),
              ObservedArm((B,), 0, f"{evidence_id}/b"))), **changes)


def filters(**changes):
    return {"allowed_train_task_ids": {"train-1"}, "excluded_source_groups": set(),
            "executor_id": "executor-a", "environment_id": "environment-a",
            "skill_versions": (A, B), **changes}


@pytest.mark.parametrize("changes", [
    {"task_id": "unallowed-task"}, {"source_group": "held-database"},
    {"split": "internal_dev"}, {"split": "official_dev"}, {"split": "test"},
    {"executor_id": "executor-b"}, {"environment_id": "environment-b"}, {"seed": 1},
])
def test_wrong_scope_drops_entire_episode(changes):
    graph = SkillEvidenceGraph()
    graph.append(episode("allowed"))
    graph.append(episode("excluded", **changes))
    result = graph.view(**filters(excluded_source_groups={"held-database"}, seed=0))
    assert tuple(edge.evidence_id for edge in result) == ("allowed",)
    assert len(result[0].arms) == 2


def test_exact_document_and_lesson_version_filter_preserves_missingness():
    graph = SkillEvidenceGraph()
    graph.append(episode())
    wrong_doc = replace(A, document_sha256="f" * 64)
    assert graph.single_arm_observations(**filters(skill_versions=(wrong_doc,))) == ()
    assert graph.view(**filters(skill_versions=(replace(B, lesson_sha256=None),))) == ()
    assert graph.view(**filters(skill_versions=(replace(B, lesson_sha256="f" * 64),))) == ()
    rows = graph.single_arm_observations(**filters(skill_versions=(B,)))
    assert len(rows) == 1 and rows[0]["skill_id"] == "b" and rows[0]["outcome"] == 0
    assert rows[0]["trace_id"] == "e/b" and rows[0]["context_sha256"] == "e" * 64
    assert len(graph.history[0].arms) == 2
    assert graph.view(**filters(allowed_train_task_ids=set())) == ()


def test_joint_reward_never_becomes_individual_credit_or_partial_bundle():
    graph = SkillEvidenceGraph()
    joint = ObservedArm((A, B), 1, "joint-trace")
    control = ObservedArm((), 1, "control-trace")
    graph.append(episode(arms=(joint, control, ObservedArm((A,), 0, "single-trace"))))
    assert graph.view(**filters())[0].arms == (joint, control, ObservedArm((A,), 0, "single-trace"))
    rows = graph.single_arm_observations(**filters())
    assert len(rows) == 1 and rows[0]["skill_id"] == "a" and rows[0]["outcome"] == 0
    filtered = graph.view(**filters(skill_versions=(A,)))[0]
    assert len(filtered.arms) == 2 and all(len(arm.skills) <= 1 for arm in filtered.arms)
    assert {arm.trace_id for arm in filtered.arms} == {"control-trace", "single-trace"}
    # Retained control is still an observed no-skill arm, not a synthetic zero.
    assert graph.view(**filters(skill_versions=()))[0].arms == (control,)


def test_duplicate_rejection_is_atomic_and_deactivation_does_not_release_trace():
    graph = SkillEvidenceGraph()
    graph.append(episode())
    before = graph.to_json()
    for duplicate in (episode(), episode("new", arms=(ObservedArm((A,), 0, "e/a"),))):
        with pytest.raises(ValueError, match="Duplicate"):
            graph.append(duplicate)
        assert graph.to_json() == before
    graph.deactivate("e", "invalid score under audit")
    with pytest.raises(ValueError, match="Duplicate trace"):
        graph.append(episode("reused", arms=(ObservedArm((A,), 1, "e/a"),)))
    with pytest.raises(ValueError, match="Duplicate trace"):
        episode(arms=(ObservedArm((A,), 0, "shared"), ObservedArm((B,), 1, "shared")))


def test_append_only_rollback_replays_and_preserves_canonical_hash():
    graph = SkillEvidenceGraph()
    graph.append(episode())
    original_view, original_hash = graph.view(**filters()), graph.snapshot_sha256
    graph.deactivate("e", "score provenance temporarily unavailable")
    assert graph.view(**filters()) == ()
    restored = SkillEvidenceGraph.from_json(graph.to_json())
    assert restored.view(**filters()) == ()
    restored.reactivate("e", "original trace and score reverified")
    assert restored.view(**filters()) == original_view
    assert restored.snapshot_sha256 != original_hash  # rollback preserves its audit trail
    assert len(restored.history) == 3 and isinstance(restored.history[-1], EvidenceStatus)
    restored.freeze()
    decoded = SkillEvidenceGraph.from_json(json.dumps(json.loads(restored.to_json()), indent=2))
    assert decoded.to_json() == restored.to_json()
    assert decoded.history == restored.history and decoded.is_frozen
    assert decoded.snapshot_sha256 == hashlib.sha256(decoded.to_json().encode()).hexdigest()


def test_frozen_reads_and_returned_records_cannot_mutate_stored_evidence():
    graph = SkillEvidenceGraph()
    graph.append(episode())
    assert graph.freeze() is graph
    snapshot = graph.to_json()
    read = graph.view(**filters())
    with pytest.raises(FrozenInstanceError):
        read[0].task_id = "changed"
    with pytest.raises(FrozenInstanceError):
        read[0].arms[0].outcome = 0
    row = graph.single_arm_observations(**filters())[0]
    row["outcome"] = 0
    assert graph.single_arm_observations(**filters())[0]["outcome"] == 1
    for action in (lambda: graph.append(episode("new")),
                   lambda: graph.deactivate("e", "reason"),
                   lambda: graph.reactivate("e", "reason")):
        with pytest.raises(RuntimeError, match="Frozen"):
            action()
    assert graph.to_json() == snapshot


@pytest.mark.parametrize("outcome", [None, -.1, 2, "1", float("nan")])
def test_missing_or_nonbinary_outcomes_are_rejected(outcome):
    with pytest.raises(ValueError, match="observed binary"):
        ObservedArm((A,), outcome, "trace")


def test_invalid_ids_versions_statuses_and_filters_fail_closed():
    with pytest.raises(ValueError):
        SkillRef("a", "not-a-hash")
    with pytest.raises(ValueError):
        episode(seed=True)
    with pytest.raises(ValueError):
        episode(context_sha256="")
    graph = SkillEvidenceGraph()
    graph.append(episode())
    for action in (lambda: graph.deactivate("e", ""),
                   lambda: graph.deactivate("missing", "reason"),
                   lambda: graph.reactivate("e", "already active"),
                   lambda: graph.view(**filters(allowed_train_task_ids="train-1")),
                   lambda: graph.view(**filters(skill_versions=("a",)))):
        with pytest.raises(ValueError):
            action()
    assert len(graph.history) == 1


def test_tampered_snapshot_cannot_bypass_replay_validation():
    graph = SkillEvidenceGraph()
    graph.append(episode())
    snapshot = json.loads(graph.to_json())
    snapshot["events"].append(snapshot["events"][0])
    with pytest.raises(ValueError, match="Duplicate evidence"):
        SkillEvidenceGraph.from_json(json.dumps(snapshot))
    with pytest.raises(ValueError, match="Duplicate JSON"):
        SkillEvidenceGraph.from_json('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError):
        SkillEvidenceGraph.from_json('{"schema_version":true,"frozen":false,"events":[]}')


def test_build_filter_and_snapshot_have_no_file_or_network_side_effects(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Evidence graph must use caller-supplied objects only")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)
    graph = SkillEvidenceGraph()
    graph.append(episode())
    graph.deactivate("e", "temporary")
    graph.reactivate("e", "reverified")
    restored = SkillEvidenceGraph.from_json(graph.freeze().to_json())
    assert len(restored.single_arm_observations(**filters())) == 2
