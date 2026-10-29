from dataclasses import replace

import pytest
import torch

from skill_experience.retrieval.skill_evidence_graph import EpisodeEvidence, ObservedArm, SkillRef
from skill_experience.retrieval.task_key_memory import TaskKeyOptMemory, task_key_gradient


def memory(n=8, **config):
    refs = tuple(SkillRef(f's{i}', f'{i:064x}') for i in range(n))
    return TaskKeyOptMemory(torch.zeros(n, 2), refs, executor_id='executor', environment_id='env', **config)


def retrieve(m, event='e', query=None, **kw):
    return m.retrieve(torch.tensor([1., 0.]) if query is None else query,
        evidence_id=event, task_id='task-'+event, source_group='group-'+event,
        query_sha256='a'*64, context_sha256='b'*64, seed=0, **kw)


def episode(slate, labels):
    return EpisodeEvidence(evidence_id=slate.evidence_id, task_id=slate.task_id,
        source_group=slate.source_group, query_sha256=slate.query_sha256,
        context_sha256=slate.context_sha256, executor_id=slate.executor_id,
        environment_id=slate.environment_id, seed=slate.seed, split='train',
        arms=tuple(ObservedArm((slate.candidates[i],), y, f'{slate.evidence_id}/{i}')
                   for i, y in labels.items()))


def test_joint_direction_equals_autograd_expected_utility():
    scores = torch.tensor([.1, -.2, .05, .4], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([1., 0., float('nan'), 1.], dtype=torch.float64)
    seen = torch.tensor([True, True, False, True])
    utility = ((scores[seen]/.3).softmax(0)*y[seen]).sum()
    actual = torch.autograd.grad(utility, scores)[0]*.02
    expected = task_key_gradient(scores.detach(), y, seen, learning_rate=.02, temperature=.3, radius=1.)
    torch.testing.assert_close(expected, actual)
    assert abs(float(expected.sum())) < 1e-12
    assert expected[2] == 0


def test_top8_updates_jointly_without_touching_bank_tail():
    m = memory(10)
    slate = retrieve(m)
    assert len(slate.candidates) == 8
    update = m.record(slate, episode(slate, {i: int(i == 0) for i in range(8)}))
    alpha = torch.tensor(update.coefficients)
    assert alpha[0] > 0 and (alpha[1:] < 0).all()
    assert alpha.norm() <= .050001 and abs(float(alpha.sum())) < 1e-7
    scores, details = m.score(torch.tensor([1., 0.]), return_details=True)
    assert scores[0] > 0 and (scores[1:8] < 0).all() and (scores[8:] == 0).all()
    assert len(details['weights']) == 1
    assert scores.norm() <= .050001


@pytest.mark.parametrize('labels', [{0: 0}, {0: 1}, {0: 0, 1: 0}, {0: 1, 1: 1}])
def test_tied_or_singleton_feedback_has_no_relative_credit(labels):
    m = memory()
    slate = retrieve(m)
    update = m.record(slate, episode(slate, labels))
    assert update.observed_count == len(labels)
    assert not any(update.coefficients)
    assert torch.equal(m.score(torch.tensor([1., 0.])), torch.zeros(8))


def test_missing_arms_and_bundle_reward_are_not_failures():
    m = memory()
    s = retrieve(m)
    e = episode(s, {0: 1, 2: 0})
    e = replace(e, arms=(*e.arms, ObservedArm((s.candidates[3], s.candidates[4]), 0, 'bundle')))
    update = m.record(s, e)
    assert update.coefficients[0] > 0 and update.coefficients[2] < 0
    assert all(update.coefficients[i] == 0 for i in [1, 3, 4, 5, 6, 7])


@pytest.mark.parametrize('field,value', [('executor_id','wrong'),('seed',1),('source_group','wrong'),
                                         ('context_sha256','c'*64),('split','test')])
def test_wrong_context_is_rejected_atomically(field, value):
    m = memory()
    s = retrieve(m)
    before = m.snapshot()
    with pytest.raises(ValueError):
        m.record(s, replace(episode(s, {0:1, 1:0}), **{field:value}))
    assert m.snapshot() == before


def test_future_no_feedback_duplicate_and_retraction():
    m = memory()
    s = retrieve(m)
    q = torch.tensor([1., 0.])
    assert torch.equal(m.score(q), torch.zeros(8))
    m.record(s, episode(s, {0:1, 1:0}))
    after = m.score(q)
    assert after[0] > 0
    with pytest.raises(ValueError):
        m.record(s, episode(s, {0:1, 1:0}))
    assert torch.equal(m.score(q, excluded_source_groups=['group-e']), torch.zeros(8))
    assert torch.equal(m.score(q, allowed_task_ids=[]), torch.zeros(8))
    m.deactivate('e', 'withdraw incorrect observation')
    assert torch.equal(m.score(q), torch.zeros(8))
    m.reactivate('e', 'verified original observation')
    torch.testing.assert_close(after, m.score(q))


def test_altered_slate_and_skill_version_are_rejected():
    m = memory()
    s = retrieve(m)
    with pytest.raises(ValueError):
        m.record(replace(s, scores=(9.,)*8), episode(s, {0:1,1:0}))
    e = episode(s, {0:1,1:0})
    with pytest.raises(ValueError):
        m.record(s, replace(e, arms=(ObservedArm((SkillRef('s0','f'*64),), 1,'changed'),)))


def test_conditional_event_reading_preserves_opposite_contexts_that_global_average_erases():
    m = memory(2, similarity_floor=.8, memory_top_m=1)
    queries = torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    labels = [0,0,1,1]
    for i, (q, target) in enumerate(zip(queries, labels, strict=True)):
        s = retrieve(m, event=str(i), query=q)
        m.record(s, episode(s, {0:int(target==0), 1:int(target==1)}))
    keys, bias = m.compile_global()
    torch.testing.assert_close(keys, torch.zeros_like(keys), atol=1e-7, rtol=0)
    for q, target in zip(queries, labels, strict=True):
        assert int(m.score(q).argmax()) == target
        torch.testing.assert_close(m.score(q, mode='global'), keys @ q+bias)
    # Synthetic capacity check only; it is not a benchmark success claim.
    q = torch.tensor([2**-.5,2**-.5])
    torch.testing.assert_close(m.score(q), torch.zeros(2))


def test_archived_candidates_require_frozen_selection_receipt():
    m = memory(10)
    selected = (m.refs[8], m.refs[9])
    with pytest.raises(ValueError):
        retrieve(m, candidate_refs=selected)
    s = retrieve(m, candidate_refs=selected, selection_sha256='c'*64)
    assert s.candidates == selected and s.candidate_source == 'historical_receipt'
    m.record(s, episode(s, {0:1,1:0}))
    assert m.score(torch.tensor([1.,0.]))[8] > 0
