import pytest
import torch

from skill_experience.retrieval.evidence_candidate_router import EvidenceCandidateRouter
from skill_experience.retrieval.skill_evidence_graph import EpisodeEvidence, ObservedArm, SkillRef
from skill_experience.retrieval.task_key_memory import TaskKeyOptMemory


def fixture():
    refs = tuple(SkillRef(f"s{i}", f"{i:064x}") for i in range(5))
    memory = TaskKeyOptMemory(torch.zeros(5, 2), refs,
                              executor_id="actor", environment_id="env")
    query = torch.tensor([1., 0.])
    slate = memory.retrieve(query, evidence_id="past", task_id="past-task",
        source_group="past-group", query_sha256="a"*64, context_sha256="b"*64,
        seed=0, top_k=5)
    episode = EpisodeEvidence("past", "past-task", "past-group", "a"*64, "train",
        "actor", "env", 0, "b"*64,
        (ObservedArm((refs[0],), 0, "trace-0"), ObservedArm((refs[4],), 1, "trace-4")))
    memory.record(slate, episode)
    router = EvidenceCandidateRouter(memory, candidate_budget=3,
                                      protected_semantic=2, max_expansion=1)
    scores = torch.tensor([.04, .03, .02, .01, 0.])
    return memory, router, query, scores


def call(router, q, scores, **kw):
    args = dict(current_task_id="future", current_source_group="future-group",
                allowed_history_task_ids=["past-task"])
    args.update(kw)
    return router.route(q, scores, **args)


def test_pool_expands_to_observed_winner_not_unknown_slate_member():
    _, router, q, scores = fixture()
    d = call(router, q, scores)
    assert [x.skill_id for x in d.candidates] == ["s0", "s1", "s4"]
    assert [x.skill_id for x in d.added] == ["s4"]
    assert d.ranked[0].skill_id == "s4"
    assert d.evidence_ids == ("past",)
    assert sum(d.event_weights) < 1


def test_bias_control_has_identical_pool_and_no_extra_residual():
    _, router, q, scores = fixture()
    keyopt = call(router, q, scores)
    control = call(router, q, scores, use_keyopt=False)
    assert control.candidates == keyopt.candidates
    assert control.ranked[0].skill_id == "s0"
    assert not any(control.corrections)


@pytest.mark.parametrize("kw", [dict(allowed_history_task_ids=[]),
    dict(current_task_id="past-task"), dict(current_source_group="past-group"),
    dict(excluded_source_groups=["past-group"])])
def test_permissions_remove_whole_event_and_restore_semantic_candidates(kw):
    _, router, q, scores = fixture()
    d = call(router, q, scores, **kw)
    assert d.candidates == d.semantic_candidates
    assert not d.evidence_ids and not d.added and not any(d.corrections)


def test_retraction_and_candidate_only_ablation():
    memory, router, q, scores = fixture()
    d = call(router, q, scores, expand=False)
    assert d.candidates == d.semantic_candidates
    assert any(d.corrections)
    memory.deactivate("past", "withdrawn")
    d = call(router, q, scores)
    assert not d.evidence_ids and not any(d.corrections)


def test_invalid_allowlist_or_bank_scores_rejected():
    _, router, q, scores = fixture()
    with pytest.raises(ValueError):
        call(router, q, scores, allowed_history_task_ids=None)
    with pytest.raises(ValueError):
        call(router, q, scores[:3])
