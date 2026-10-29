import json

import pytest

from skill_experience.retrieval.evidence_agent import (
    compatible,
    messages,
    parse,
    select_events,
    shuffled_events,
)


def fixture():
    public = dict(
        task_id="new",
        question="New question",
        query=[1.0, 0.0],
        baseline_skill_id="a",
        candidates=[
            dict(skill_id="a", document="Alpha method.", document_sha256="va"),
            dict(skill_id="b", document="Beta method.", document_sha256="vb"),
        ],
    )
    events = [
        dict(
            event_id=str(i),
            task_id=str(i),
            source_group=str(i),
            question="old " + str(i),
            query=[1.0, 0.0],
            versions={"a": "va", "b": "vb"},
            outcomes={"a": i % 2, "b": 1 - i % 2},
            executor="old-executor",
            environment="old-tools",
        )
        for i in range(10)
    ]
    return public, events


def test_excludes_self_question_and_version():
    p, e = fixture()
    e[0]["task_id"] = p["task_id"]
    e[1]["question"] = " NEW  QUESTION "
    e[2]["versions"] = {"a": "wrong", "b": "vb"}
    assert {x["event_id"] for x in compatible(p, e)} == {str(i) for i in range(3, 10)}


def test_unknown_is_not_failure():
    p, e = fixture()
    e[0]["outcomes"] = {"a": 1}
    with pytest.raises(ValueError, match="Incomplete"):
        compatible(p, e)


def test_balanced_evidence_and_shuffle_conservation():
    p, e = fixture()
    chosen, count = select_events(p, e)
    assert count == 10 and len(chosen) == 4
    assert sum(x["outcomes"]["a"] for x in chosen) == 2
    s = shuffled_events(e)
    assert sum(v["outcomes"]["a"] for v in s.values()) == 5
    assert all(s[x["event_id"]]["versions"] == x["versions"] for x in e)


def test_none_has_no_history_and_foreign_quote_falls_back():
    p, e = fixture()
    prompt = messages(p, e, "none")
    assert json.loads(prompt[-1]["content"])["history"] == []
    value = dict(skill_id="b", quote="Beta method.", evidence_ids=[], reason="applicable")
    assert parse(json.dumps(value), prompt)["valid"]
    value["quote"] = "Alpha method."
    assert parse(json.dumps(value), prompt)["skill_id"] == "a"
    assert not parse(json.dumps(value), prompt)["valid"]


def test_wrong_evidence_and_malformed_fall_back():
    p, e = fixture()
    prompt = messages(p, e[:2], "real")
    value = dict(skill_id="b", quote="Beta method.", evidence_ids=["missing"], reason="a")
    assert not parse(json.dumps(value), prompt)["valid"]
    assert not parse("null", prompt)["valid"]


def test_duplicate_source_counts_once():
    p, e = fixture()
    e[1]["source_group"] = e[0]["source_group"]
    assert len(compatible(p, e)) == 9


def test_invalid_reader_keeps_keyopt_proposal_not_semantic_rank():
    p, e = fixture()
    p['proposed_skill_id'] = 'b'
    prompt = messages(p, e, 'real')
    out = parse('not JSON', prompt)
    assert not out['valid']
    assert out['skill_id'] == 'b'
