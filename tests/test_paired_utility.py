"""Matched outcomes, not coincident failures, supervise conditional updates."""

import pytest
import torch

from skill_experience.retrieval import paired_utility
from skill_experience.retrieval.paired_utility import (
    direct_pair_prototype,
    directional_pairs,
    directional_ranking_loss,
    fit_bias_plus_prototype,
    fit_bounded_key,
)


def test_only_opposite_outcomes_become_task_balanced_labels():
    candidates = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    outcomes = torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 1., 1.]])
    task, winner, loser, weight = directional_pairs(candidates, outcomes)
    assert len(task) == 4
    assert set(task.tolist()) == {0, 2}
    assert not (task == 1).any()
    assert torch.isclose(weight[task == 0].sum(), torch.tensor(.5))
    assert torch.isclose(weight[task == 2].sum(), torch.tensor(.5))
    assert {(int(w), int(loser_id)) for w, loser_id in zip(winner, loser)} == {
        (0, 1), (0, 2), (1, 0), (2, 0)}


def test_tied_outcomes_do_not_push_key_or_bias():
    queries = torch.tensor([[1., 0.], [-1., 0.], [0., 1.]])
    base = torch.tensor([[.2, .0], [.0, .2]])
    candidates = torch.tensor([[0, 1]] * 3)
    outcomes = torch.tensor([[0., 0.], [1., 1.], [0., 0.]])
    pairs = directional_pairs(candidates, outcomes)
    scores = (queries @ base.T).requires_grad_()
    loss = directional_ranking_loss(scores, pairs)
    loss.backward()
    assert torch.equal(scores.grad, torch.zeros_like(scores))
    for mode in ("bias", "pca8", "linear"):
        keys, bias, audit = fit_bounded_key(
            queries, base, candidates, outcomes, mode=mode, radius=.3,
            steps=30, learning_rate=.1, keep=.01)
        assert torch.equal(keys, base)
        assert torch.equal(bias, torch.zeros(2))
        assert audit["directional_tasks"] == 0
    keys, bias, audit = direct_pair_prototype(
        queries, base, candidates, outcomes, radius=.3)
    assert torch.equal(keys, base) and torch.equal(bias, torch.zeros(2))
    assert audit["directional_pairs"] == 0
    keys, bias, audit = fit_bias_plus_prototype(
        queries, base, candidates, outcomes, radius=.3, bias_fraction=.5,
        steps=30, learning_rate=.1, keep=.01)
    assert torch.equal(keys, base) and torch.equal(bias, torch.zeros(2))
    assert audit["directional_pairs"] == 0


def test_conditional_key_can_express_opposite_preferences_bias_cannot():
    queries = torch.tensor([[1., 0.], [-1., 0.]] * 12)
    base = torch.zeros((2, 2))
    candidates = torch.tensor([[0, 1]] * len(queries))
    outcomes = torch.tensor([[1., 0.], [0., 1.]] * 12)
    bias_keys, bias, _ = fit_bounded_key(
        queries, base, candidates, outcomes, mode="bias", radius=.5,
        steps=100, learning_rate=.05, keep=.001)
    key, offset, audit = fit_bounded_key(
        queries, base, candidates, outcomes, mode="pca8", radius=.5,
        steps=100, learning_rate=.05, keep=.001)
    baseline_choice = (queries @ bias_keys.T + bias).argmax(1)
    conditional_choice = (queries @ key.T + offset).argmax(1)
    expected = outcomes.argmax(1)
    assert (baseline_choice == expected).sum() <= 12
    assert (conditional_choice == expected).sum() == len(queries)
    assert audit["directional_tasks"] == len(queries)
    assert audit["directional_pairs"] == len(queries)
    linear_key, linear_bias, _ = fit_bounded_key(
        queries, base, candidates, outcomes, mode="linear", radius=.5,
        steps=100, learning_rate=.05, keep=.001)
    assert ((queries @ linear_key.T + linear_bias).argmax(1) == expected).all()


def test_bias_plus_prototype_shares_one_radius_and_both_feedback_components():
    queries = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [-1., 0.]])
    base = torch.zeros((2, 2))
    candidates = torch.tensor([[0, 1]] * len(queries))
    outcomes = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.]])
    radius = .4
    keys, bias, audit = fit_bias_plus_prototype(
        queries, base, candidates, outcomes, radius=radius, bias_fraction=.25,
        steps=80, learning_rate=.05, keep=.001)
    _, expected_bias, _ = fit_bounded_key(
        queries, base, candidates, outcomes, mode="bias", radius=radius * .25,
        steps=80, learning_rate=.05, keep=.001)
    expected_keys, _, _ = direct_pair_prototype(
        queries, base, candidates, outcomes, radius=radius * .75)
    assert torch.allclose(keys, expected_keys)
    assert torch.allclose(bias, expected_bias)
    assert torch.any((keys - base).abs() > 0)
    assert torch.any(bias.abs() > 0)
    assert torch.all((keys - base).norm(dim=1) + bias.abs() <= radius + 1e-6)
    assert audit["bias_radius"] == radius * .25
    assert audit["prototype_radius"] == radius * .75
    assert audit["directional_tasks"] == len(queries)
    with pytest.raises(ValueError, match="interior bias fraction"):
        fit_bias_plus_prototype(
            queries, base, candidates, outcomes, radius=radius,
            bias_fraction=1., steps=10, learning_rate=.05, keep=.001)


def test_global_radius_prototype_preserves_relative_strength_and_uses_budget():
    queries = torch.tensor([[1., 0.], [0., 1.]])
    base = torch.zeros((3, 2))
    candidates = torch.arange(3).expand(2, -1)
    outcomes = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    legacy_small, _, _ = direct_pair_prototype(
        queries, base, candidates, outcomes, radius=.8)
    legacy_large, _, _ = direct_pair_prototype(
        queries, base, candidates, outcomes, radius=1.6, scale_mode="clip")
    assert torch.equal(legacy_small, legacy_large)
    scaled_small, _, small_audit = direct_pair_prototype(
        queries, base, candidates, outcomes, radius=.8, scale_mode="global_radius")
    scaled_large, _, large_audit = direct_pair_prototype(
        queries, base, candidates, outcomes, radius=1.6, scale_mode="global_radius")
    assert not torch.equal(scaled_small, scaled_large)
    assert torch.allclose(scaled_large, 2 * scaled_small)
    assert torch.isclose(scaled_small.norm(dim=1).max(), torch.tensor(.8))
    assert torch.isclose(scaled_large.norm(dim=1).max(), torch.tensor(1.6))
    assert torch.all(scaled_small.norm(dim=1) <= .8 + 1e-6)
    assert torch.all(scaled_large.norm(dim=1) <= 1.6 + 1e-6)
    assert small_audit["raw_max_norm"] == large_audit["raw_max_norm"]
    assert small_audit["global_scale"] > 1
    assert large_audit["global_scale"] == pytest.approx(2 * small_audit["global_scale"])
    assert small_audit["updated_max_norm"] == pytest.approx(.8)


def test_global_radius_zero_evidence_stays_zero_and_legacy_default_is_unchanged():
    queries = torch.tensor([[1., 0.], [0., 1.]])
    base = torch.tensor([[.2, .0], [.0, .2]])
    candidates = torch.arange(2).expand(2, -1)
    ties = torch.tensor([[0., 0.], [1., 1.]])
    keys, bias, audit = direct_pair_prototype(
        queries, base, candidates, ties, radius=.8, scale_mode="global_radius")
    assert torch.equal(keys, base)
    assert torch.equal(bias, torch.zeros(2))
    assert audit["raw_max_norm"] == 0
    assert audit["global_scale"] == 0
    assert audit["updated_max_norm"] == 0
    with pytest.raises(ValueError, match="Invalid prototype"):
        direct_pair_prototype(queries, base, candidates, ties,
                              radius=.8, scale_mode="unknown")


def test_scaled_combination_fits_bias_after_updated_keys(monkeypatch):
    queries = torch.tensor([[1., 0.], [0., 1.]])
    base = torch.zeros((3, 2))
    candidates = torch.arange(3).expand(2, -1)
    outcomes = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    expected_keys, _, _ = direct_pair_prototype(
        queries, base, candidates, outcomes, radius=.3,
        scale_mode="global_radius")
    fitted_bases = []

    def record_bias_fit(_queries, base_keys, _candidates, _success, **kwargs):
        assert kwargs["mode"] == "bias"
        assert kwargs["radius"] == pytest.approx(.1)
        fitted_bases.append(base_keys.clone())
        return base_keys.clone(), base_keys.new_zeros(len(base_keys)), {
            "directional_tasks": 2, "directional_pairs": 4,
            "supported_skills": 3}

    monkeypatch.setattr(paired_utility, "fit_bounded_key", record_bias_fit)
    keys, bias, audit = fit_bias_plus_prototype(
        queries, base, candidates, outcomes, radius=.4, bias_fraction=.25,
        steps=5, learning_rate=.05, keep=.01,
        prototype_mode="scaled_then_bias")
    assert len(fitted_bases) == 1
    assert torch.allclose(fitted_bases[0], expected_keys)
    assert torch.allclose(keys, expected_keys)
    assert torch.equal(bias, torch.zeros(3))
    assert audit["prototype_mode"] == "scaled_then_bias"
    assert audit["prototype_raw_max_norm"] > 0
    assert audit["prototype_updated_max_norm"] == pytest.approx(.3)


def test_scaled_combination_respects_shared_budget_and_legacy_explicit_matches_default():
    queries = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [-1., 0.]])
    base = torch.zeros((2, 2))
    candidates = torch.arange(2).expand(4, -1)
    outcomes = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.]])
    arguments = dict(radius=.4, bias_fraction=.25, steps=80,
                     learning_rate=.05, keep=.001)
    old_keys, old_bias, _ = fit_bias_plus_prototype(
        queries, base, candidates, outcomes, **arguments)
    explicit_keys, explicit_bias, explicit_audit = fit_bias_plus_prototype(
        queries, base, candidates, outcomes, **arguments,
        prototype_mode="legacy")
    assert torch.equal(old_keys, explicit_keys)
    assert torch.equal(old_bias, explicit_bias)
    assert explicit_audit["prototype_mode"] == "legacy"
    keys, bias, audit = fit_bias_plus_prototype(
        queries, base, candidates, outcomes, **arguments,
        prototype_mode="scaled_then_bias")
    assert torch.all((keys - base).norm(dim=1) + bias.abs() <= .4 + 1e-6)
    assert audit["prototype_updated_max_norm"] == pytest.approx(.3)
    with pytest.raises(ValueError, match="valid mode"):
        fit_bias_plus_prototype(
            queries, base, candidates, outcomes, **arguments,
            prototype_mode="unknown")
