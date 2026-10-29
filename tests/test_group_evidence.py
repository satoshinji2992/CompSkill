import copy

import pytest

from skill_experience.retrieval.group_evidence import GroupEvidenceMemory, SkillGroup
from skill_experience.retrieval.skill_evidence_graph import SkillRef


def group(names, version="a" * 64):
    return SkillGroup(tuple(SkillRef(n, version) for n in names),
                      tuple("anchor" if i == 0 else "support" for i in range(len(names))), "roles-v1")


def fixture(split="train", repeat=0):
    memory = GroupEvidenceMemory(dimension=2)
    ab, ac = group("ab"), group("ac")
    slate = memory.register(event_id="e", task_id="past", source_group="source-past",
        executor_id="ds", environment_id="sb", repeat=repeat, step=0, query=[1., 0.],
        groups=[ab, ac], scores=[.4, .4])
    memory.commit(slate, observed={ab.identity: dict(outcome=1, trace_id="ab"),
                  ac.identity: dict(outcome=0, trace_id="ac")}, available_step=1, split=split)
    return memory, ab, ac


def read(memory, groups, **kwargs):
    args = dict(allowed_history_task_ids=["past"], current_task_id="future",
        current_source_group="source-future", executor_id="ds", environment_id="sb", repeat=0, step=2)
    args.update(kwargs)
    return memory.corrections(groups, [1., 0.], **args)


def test_group_feedback_does_not_credit_constituents_or_unobserved_groups():
    mem, ab, ac = fixture()
    a, abc = group("a"), group("abc")
    scores = read(mem, [ab, ac, a, abc])
    assert scores[ab.identity]["keyopt"] > 0 > scores[ac.identity]["keyopt"]
    assert scores[a.identity]["bias"] == scores[a.identity]["keyopt"] == 0
    assert scores[abc.identity]["bias"] == 0


@pytest.mark.parametrize("options", [dict(current_task_id="past"), dict(current_source_group="source-past"),
    dict(allowed_history_task_ids=[]), dict(executor_id="qwen"), dict(environment_id="elsewhere"), dict(step=0)])
def test_current_task_source_executor_and_future_results_cannot_leak(options):
    mem, ab, _ = fixture()
    assert read(mem, [ab], **options)[ab.identity]["support_tasks"] == 0


def test_versions_roles_order_and_repeat_are_bound():
    mem, ab, _ = fixture(split="online")
    changed = [group("ab", "b" * 64), group("ba"),
               SkillGroup(ab.members, ("support", "anchor"), "roles-v1")]
    assert all(v["support_tasks"] == 0 for v in read(mem, changed).values())
    assert read(mem, [ab], repeat=1)[ab.identity]["support_tasks"] == 0


def test_pair_projection_can_infer_but_exact_event_nodes_match_hypergraph():
    mem, ab, _ = fixture()
    abc = group("abd")
    assert read(mem, [abc], view="pairwise")[abc.identity]["bias"] != 0
    assert read(mem, [ab, abc]) == read(mem, [ab, abc], view="event_nodes")


def test_roundtrip_and_duplicate_trace_rejection_are_atomic():
    mem, ab, ac = fixture()
    restored = GroupEvidenceMemory.restore(copy.deepcopy(mem.export()))
    assert restored.fingerprint == mem.fingerprint
    assert read(restored, [ab, ac]) == read(mem, [ab, ac])
    slate = mem.register(event_id="second", task_id="another", source_group="another",
        executor_id="ds", environment_id="sb", repeat=0, step=3, query=[1., 0.], groups=[ab], scores=[0.])
    before = mem.fingerprint
    with pytest.raises(ValueError, match="trace"):
        mem.commit(slate, observed={ab.identity: dict(outcome=1, trace_id="ab")}, available_step=4)
    assert mem.fingerprint == before


def test_single_observation_updates_group_reliability_but_not_relative_keyopt():
    mem = GroupEvidenceMemory(dimension=2)
    ab, ac = group("ab"), group("ac")
    slate = mem.register(event_id="e", task_id="past", source_group="source-past",
        executor_id="ds", environment_id="sb", repeat=0, step=0, query=[1., 0.],
        groups=[ab, ac], scores=[0., 0.])
    event = mem.commit(slate, observed={ab.identity: dict(outcome=1, trace_id="one")}, available_step=1)
    assert event["coefficients"] == [0., 0.]
    result = read(mem, [ab, ac])
    assert result[ab.identity]["bias"] > 0
    assert result[ac.identity]["support_tasks"] == 0


def test_export_and_commit_receipts_cannot_mutate_memory():
    mem, ab, _ = fixture()
    before = mem.fingerprint
    data = mem.export()
    data["events"][0]["observed"][ab.identity]["outcome"] = 0
    assert mem.fingerprint == before


def test_repeated_task_is_not_counted_as_more_independent_support():
    mem, ab, ac = fixture()
    before = read(mem, [ab, ac])
    slate = mem.register(event_id="repeat", task_id="past", source_group="source-past",
        executor_id="ds", environment_id="sb", repeat=1, step=0, query=[1., 0.],
        groups=[ab, ac], scores=[.4, .4])
    mem.commit(slate, observed={ab.identity: dict(outcome=1, trace_id="ab2"),
               ac.identity: dict(outcome=0, trace_id="ac2")}, available_step=1)
    after = read(mem, [ab, ac])
    assert after[ab.identity]["support_tasks"] == 1
    assert after[ab.identity]["bias"] == before[ab.identity]["bias"]
    assert after[ab.identity]["keyopt"] == before[ab.identity]["keyopt"]
