import pytest
import torch

from skill_experience.retrieval.graph_anchored_keyopt import (
    fit_graph_anchored_keyopt,
    validate_memory_version,
)
from skill_experience.retrieval.skill_evidence_graph import EpisodeEvidence, ObservedArm, SkillRef
from skill_experience.retrieval.task_key_memory import TaskKeyOptMemory


def example():
    refs = tuple(SkillRef(f"s{i}", f"{i:064x}") for i in range(4))
    memory = TaskKeyOptMemory(torch.zeros(4, 2), refs, executor_id="e", environment_id="env",
                              bias=torch.tensor([.05, -.05, 0., 0.]))
    for i, q in enumerate((torch.tensor([1., 0.]), torch.tensor([0., 1.]))):
        slate = memory.retrieve(q, evidence_id=f"event{i}", task_id=f"task{i}",
            source_group=f"group{i}", query_sha256="a"*64, context_sha256="b"*64,
            seed=17, top_k=4, mode="none")
        arms = (ObservedArm((refs[0],), 1-i, f"{i}/0"),
                ObservedArm((refs[1],), i, f"{i}/1"),
                ObservedArm((refs[2], refs[3]), 1, f"{i}/bundle"))
        memory.record(slate, EpisodeEvidence(slate.evidence_id, slate.task_id,
            slate.source_group, slate.query_sha256, "train", slate.executor_id,
            slate.environment_id, 17, slate.context_sha256, arms))
    return memory


def test_zero_residual_exactly_reproduces_graph_anchor_and_filters():
    memory = example()
    index, receipt = fit_graph_anchored_keyopt(memory, allowed_train_task_ids=["task0", "task1"],
                                              excluded_source_groups=["group1"], radius=0.)
    keys, bias = memory.compile_global(allowed_task_ids=["task0"], excluded_source_groups=[])
    torch.testing.assert_close(index.keys, keys, rtol=0, atol=0)
    torch.testing.assert_close(index.bias, bias, rtol=0, atol=0)
    assert receipt["evidence_ids"] == ["event0"]
    assert receipt["trace_ids"] == ["0/0", "0/1"]
    assert receipt["observed_arms"] == 2  # Unknown/bundle members get no labels.


def test_conditional_update_is_centered_and_does_not_train_bundle_members():
    memory = example()
    index, receipt = fit_graph_anchored_keyopt(memory, allowed_train_task_ids=["task0", "task1"],
                                              radius=.2, steps=30, keep=.001)
    keys, bias = memory.compile_global(allowed_task_ids=["task0", "task1"])
    q = torch.eye(2)
    correction = index.score(q) - (q @ keys.T + bias)
    torch.testing.assert_close(correction.mean(0), torch.zeros(4), atol=1e-7, rtol=0)
    assert index.delta[:2].norm() > 0
    assert index.delta[2:].count_nonzero() == 0
    assert receipt["fitting"]["steps_completed"] == 30
    assert float(index.delta.norm(dim=1).max()) <= .200001


def test_retraction_invalidates_fitted_anchor():
    memory = example()
    _, receipt = fit_graph_anchored_keyopt(memory, allowed_train_task_ids=["task0", "task1"])
    validate_memory_version(memory, receipt)
    memory.deactivate("event0", "invalid execution evidence")
    with pytest.raises(ValueError, match="Evidence changed"):
        validate_memory_version(memory, receipt)


def test_empty_allowed_view_produces_no_update():
    memory = example()
    index, receipt = fit_graph_anchored_keyopt(memory, allowed_train_task_ids=[], radius=.2)
    keys, bias = memory.compile_global(allowed_task_ids=[])
    torch.testing.assert_close(index.keys, keys)
    torch.testing.assert_close(index.bias, bias)
    assert receipt["observed_arms"] == 0
    assert receipt["fitting"]["steps_completed"] == 0
    assert receipt["no_update"] is True


def test_empty_graph_version_still_binds_base_index():
    refs = (SkillRef("s", "a"*64),)
    memory = TaskKeyOptMemory(torch.zeros(1, 2), refs, executor_id="e", environment_id="env")
    _, receipt = fit_graph_anchored_keyopt(memory, allowed_train_task_ids=[])
    other = TaskKeyOptMemory(torch.ones(1, 2), refs, executor_id="e", environment_id="env")
    with pytest.raises(ValueError, match="Evidence changed"):
        validate_memory_version(other, receipt)
