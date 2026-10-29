import pytest
import torch

from skill_experience.retrieval.evidence_key_compiler import (
    EvidenceKeyCompiler,
    expected_utility_loss,
    observed_evidence_bank,
)
from skill_experience.retrieval.memory_compiler import support_mask


def sample():
    torch.manual_seed(17)
    q = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    y = torch.tensor([[1., 0., 1.], [0., 0., 0.], [1., 1., 1.],
                      [0., 1., 0.], [float("nan"), 1., float("nan")]])
    observed = torch.isfinite(y)
    return q, k, y, observed


@pytest.mark.parametrize("mode", ["mean", "attention"])
def test_bounded_span_and_masked_future_invariance(mode):
    q, k, y, observed = sample()
    e, d, owners = observed_evidence_bank(q, k, y, observed)
    mask = support_mask(owners, [0, 1], [2, 3, 4])
    net = EvidenceKeyCompiler(8, 16, mode)
    delta, bias, details = net(k, e, d, mask, return_details=True)
    weighted = (d * details["weights"][..., None]).sum(1)
    assert (details["weights"] >= 0).all()
    assert (details["weights"].sum(-1) <= 1.000001).all()
    assert (delta.norm(dim=-1) + bias.abs()).max() <= .300001
    for i in range(len(k)):
        if delta[i].norm() > 0:
            torch.testing.assert_close(torch.nn.functional.normalize(delta[i], dim=0),
                                       torch.nn.functional.normalize(weighted[i, :-1], dim=0))
    e[~mask], d[~mask] = float("nan"), float("nan")
    other = net(k, e, d, mask)
    torch.testing.assert_close(delta, other[0])
    torch.testing.assert_close(bias, other[1])


def test_distinct_task_budget_ties_unknown_and_single_observation():
    q, k, y, observed = sample()
    e, directions, owners = observed_evidence_bank(q, k, y, observed)
    assert (owners >= 0).sum(1).tolist() == [4, 5, 4]
    assert directions[:, 1:3].eq(0).all()
    assert directions[:, 4].eq(0).all()
    assert e[0, 4].eq(0).all()
    torch.testing.assert_close(directions[1, 0, :-1], -q[0])


@pytest.mark.parametrize("mode", ["mean", "attention"])
def test_no_evidence_and_explicit_no_update(mode):
    q, k, y, observed = sample()
    e, d, owners = observed_evidence_bank(q, k, y, observed)
    net = EvidenceKeyCompiler(8, 16, mode)
    a, b = net(k, e, d, torch.zeros_like(owners, dtype=torch.bool))
    assert a.eq(0).all() and b.eq(0).all()
    with torch.no_grad():
        net.gates.bias.fill_(-1)
    a, b = net(k, e, d, owners >= 0)
    assert a.eq(0).all() and b.eq(0).all()


@pytest.mark.parametrize("mode", ["mean", "attention"])
def test_skill_and_evidence_permutations_do_not_encode_identity(mode):
    q, k, y, observed = sample()
    e, d, owners = observed_evidence_bank(q, k, y, observed)
    net = EvidenceKeyCompiler(8, 16, mode).eval()
    a, b = net(k, e, d, owners >= 0)
    p, t = torch.tensor([2, 0, 1]), torch.tensor([3, 0, 4, 1, 2])
    c, z = net(k[p], e[p][:, t], d[p][:, t], owners[p][:, t] >= 0)
    torch.testing.assert_close(c, a[p])
    torch.testing.assert_close(z, b[p])


def test_observed_utility_loss_ignores_unknown_and_tie_preferences():
    scores = torch.tensor([[.1, .2, 999.], [.2, .3, .4]], requires_grad=True)
    y = torch.tensor([[1., 0., float("nan")], [0., 0., 0.]])
    mask = torch.isfinite(y)
    loss, parts = expected_utility_loss(scores, y, mask)
    loss.backward()
    assert scores.grad[0, 0] < 0 and scores.grad[0, 1] > 0
    assert scores.grad[0, 2] == 0 and scores.grad[1].eq(0).all()
    assert torch.isfinite(parts["pair"])


def test_gradients_train_shared_weights_and_compiled_scores_are_equivalent():
    q, k, y, observed = sample()
    e, d, owners = observed_evidence_bank(q, k, y, observed)
    net = EvidenceKeyCompiler(8, 16)
    mask = support_mask(owners, [0, 1, 2], [3, 4])
    delta, bias = net(k, e, d, mask)
    scores = q @ (k + delta).T + bias
    loss, _ = expected_utility_loss(scores[3:], y[3:], observed[3:])
    loss.backward()
    assert net.gates.weight.grad.abs().sum() > 0
    compiled = torch.cat((k + delta, bias[:, None]), -1)
    indexed = torch.cat((q, torch.ones(len(q), 1)), -1) @ compiled.T
    torch.testing.assert_close(indexed, scores)
