from dataclasses import replace

import pytest
import torch

from skill_experience.retrieval.reliability_key_compiler import (
    ReliabilityGraphCompiler,
    compiler_episode_loss,
    fit_frozen_bias_keyopt,
    fit_reliability_prior,
    masked_pairs,
)


def example():
    generator = torch.Generator().manual_seed(7)
    keys = torch.nn.functional.normalize(torch.randn(4, 8, generator=generator), dim=1)
    q = torch.nn.functional.normalize(torch.randn(6, 8, generator=generator), dim=1)
    y = torch.tensor([[1., 0., 1., float("nan")], [1., 0., 0., float("nan")],
                      [0., 1., 0., float("nan")], [1., 1., 1., float("nan")],
                      [float("nan"), 1., float("nan"), float("nan")],
                      [float("nan")] * 4])
    return keys, q, y, torch.isfinite(y)


def test_ragged_comparisons_balance_events_not_expanded_pairs():
    y = torch.tensor([[1., 0., float("nan")], [1., 0., 0.], [1., 1., 1.]])
    task, win, lose, weight = masked_pairs(y, torch.isfinite(y))
    assert task.tolist() == [0, 1, 1]
    assert win.tolist() == [0, 0, 0] and lose.tolist() == [1, 1, 2]
    torch.testing.assert_close(weight, torch.tensor([.5, .25, .25]))


@pytest.mark.parametrize("mode", ["flat", "pairwise", "factor", "hyperedge"])
def test_initial_prior_is_exact_and_graph_respects_mask_and_own_budget(mode):
    k, q, y, mask = example()
    prior, _ = fit_reliability_prior(k, q, y, mask, steps=5)
    net = ReliabilityGraphCompiler(8, 16, mode, radius=.07).eval()
    torch.testing.assert_close(net(k, q, y, mask, prior), prior.bias, rtol=0, atol=0)
    with torch.no_grad():
        net.head.weight.fill_(.3)
    b, details = net(k, q, y, mask, prior, return_details=True)
    assert details['correction'].abs().max() <= .07
    assert b[-1] == prior.bias[-1] == 0
    hidden = y.clone()
    hidden[~mask] = 999
    torch.testing.assert_close(net(k, q, hidden, mask, prior), b)
    # The prior is bound to values, not merely equal support counts.
    with pytest.raises(ValueError, match="different evidence"):
        net(k, q.roll(1, dims=0), y, mask, prior)


@pytest.mark.parametrize("mode", ["flat", "pairwise", "factor", "hyperedge"])
def test_permutation_equivariance_preserves_event_association(mode):
    k, q, y, mask = example()
    prior, _ = fit_reliability_prior(k, q, y, mask, steps=4)
    net = ReliabilityGraphCompiler(8, 16, mode).eval()
    with torch.no_grad():
        net.head.weight.normal_(std=.05)
    result = net(k, q, y, mask, prior)
    p, t = torch.tensor([3, 1, 0, 2]), torch.tensor([4, 2, 0, 3, 1, 5])
    other, _ = fit_reliability_prior(k[p], q[t], y[t][:, p], mask[t][:, p], steps=4)
    torch.testing.assert_close(net(k[p], q[t], y[t][:, p], mask[t][:, p], other),
                               result[p], atol=1e-6, rtol=1e-5)


def test_frozen_bias_stays_frozen_and_affine_index_is_exact():
    k, q, y, mask = example()
    prior, _ = fit_reliability_prior(k, q, y, mask, steps=5, radius=.4)
    anchor = prior.bias.clone().requires_grad_()
    untouched = anchor.detach().clone()
    result, info = fit_frozen_bias_keyopt(k, q, y, mask, anchor, steps=12, radius=.03)
    assert info['steps_completed'] == 12 and info['max_delta_norm'] <= .030001
    torch.testing.assert_close(anchor.detach(), untouched, atol=0, rtol=0)
    assert anchor.grad is None
    test = q * 1.7 + .2
    expected = test @ k.T + untouched + (test - q.mean(0)) @ result.delta.T
    torch.testing.assert_close(result.score(test), expected)
    augmented = torch.cat((test, test.new_ones(len(test), 1)), dim=1)
    torch.testing.assert_close(augmented @ result.augmented_keys().T, expected)
    torch.testing.assert_close(((q - q.mean(0)) @ result.delta.T).mean(0),
                               torch.zeros(4), atol=1e-7, rtol=0)
    assert result.delta[-1].eq(0).all()
    # Large prior does not consume the independent vector radius.
    shifted, _ = fit_frozen_bias_keyopt(k, q, y, mask, anchor + 8, steps=12, radius=.03)
    torch.testing.assert_close(shifted.delta, result.delta, atol=2e-6, rtol=1e-4)


def test_zero_radius_ties_and_empty_history_are_safe_noops():
    k, q, _, _ = example()
    for query, y in ((q, torch.ones(6, 4)), (q[:0], torch.ones(0, 4))):
        mask = torch.ones_like(y, dtype=torch.bool)
        prior, _ = fit_reliability_prior(k, query, y, mask, steps=3)
        assert prior.bias.eq(0).all()
        net = ReliabilityGraphCompiler(8, 16)
        with torch.no_grad():
            net.head.bias.fill_(10)
        assert net(k, query, y, mask, prior).eq(0).all()
        index, _ = fit_frozen_bias_keyopt(k, query, y, mask, prior.bias, steps=3)
        assert index.delta.eq(0).all()
    k, q, y, mask = example()
    prior, _ = fit_reliability_prior(k, q, y, mask, steps=4)
    index, _ = fit_frozen_bias_keyopt(k, q, y, mask, prior.bias, radius=0)
    torch.testing.assert_close(index.score(q), q @ k.T + prior.bias, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["factor", "hyperedge"])
def test_shared_episode_trains_relation_weights_and_rejects_source_overlap(mode):
    k, q, y, mask = example()
    prior, _ = fit_reliability_prior(k, q[:3], y[:3], mask[:3], steps=3)
    net = ReliabilityGraphCompiler(8, 16, mode)
    kwargs = dict(support_ids=['a', 'b', 'c'], target_ids=['d', 'e'],
                  support_groups=['train'] * 3, target_groups=['target'] * 2)
    args = (net, k, q[:3], y[:3], mask[:3], prior, q[:2], y[:2], mask[:2])
    opt = torch.optim.Adam(net.parameters(), lr=.01)
    for _ in range(3):
        loss, _, _ = compiler_episode_loss(*args, **kwargs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    parameter = net.event_node[0].weight if mode == 'factor' else net.edge_read.in_proj_weight
    assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='overlap'):
        compiler_episode_loss(*args, **{**kwargs, 'target_groups': ['train', 'target']})
    # Target outcomes affect the training loss but do not enter compiled memory.
    _, _, before = compiler_episode_loss(*args, **kwargs)
    new_args = (*args[:7], 1 - args[7], args[8])
    _, _, after = compiler_episode_loss(*new_args, **kwargs)
    torch.testing.assert_close(before, after)


def test_invalid_prior_or_observed_labels_cannot_silently_enter_training():
    k, q, y, mask = example()
    with pytest.raises(ValueError, match='binary'):
        fit_reliability_prior(k, q, y + .5, mask)
    prior, _ = fit_reliability_prior(k, q, y, mask, steps=2)
    net = ReliabilityGraphCompiler(8, 16)
    with pytest.raises(ValueError, match='support'):
        net(k, q, y, mask, replace(prior, observations=prior.observations + 1))
    with pytest.raises(ValueError):
        fit_frozen_bias_keyopt(k, q, y, mask, prior.bias, keep=float('nan'))
