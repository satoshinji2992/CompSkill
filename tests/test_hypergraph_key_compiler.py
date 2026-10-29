import pytest
import torch

from skill_experience.retrieval.evidence_key_compiler import (
    expected_utility_loss,
    observed_evidence_bank,
)
from skill_experience.retrieval.hypergraph_key_compiler import (
    HypergraphKeyCompiler,
    graph_execution_matrix,
)
from skill_experience.retrieval.skill_evidence_graph import (
    EpisodeEvidence,
    ObservedArm,
    SkillEvidenceGraph,
    SkillRef,
)


def arrays():
    torch.manual_seed(4)
    q = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    keys = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    y = torch.tensor([[1., 0., 1., float("nan")], [0., 1., 1., float("nan")],
                      [0., 0., 0., float("nan")], [1., 1., 1., float("nan")],
                      [float("nan")] * 4])
    return keys, q, y, torch.isfinite(y)


@pytest.mark.parametrize("mode", ["flat", "pairwise", "hyperedge"])
def test_permutations_masking_span_and_norm(mode):
    keys, q, y, mask = arrays()
    net = HypergraphKeyCompiler(8, 16, mode).eval()
    delta, bias, details = net(keys, q, y, mask, return_details=True)
    assert torch.isfinite(delta).all() and torch.isfinite(bias).all()
    assert (delta.norm(dim=-1) + bias.abs()).max() <= .300001
    assert delta[-1].eq(0).all() and bias[-1] == 0
    swapped = y.clone()
    swapped[~mask] = 999.
    other = net(keys, q, swapped, mask)
    torch.testing.assert_close(other[0], delta)
    torch.testing.assert_close(other[1], bias)
    p, t = torch.tensor([3, 1, 0, 2]), torch.tensor([4, 2, 0, 3, 1])
    perm = net(keys[p], q[t], y[t][:, p], mask[t][:, p])
    torch.testing.assert_close(perm[0], delta[p])
    torch.testing.assert_close(perm[1], bias[p])
    assert (details["weights"] >= 0).all()
    # Only the first two events offer nonzero directions.
    basis = q[:2].T
    recovered = basis @ torch.linalg.lstsq(basis, delta.T).solution
    torch.testing.assert_close(recovered, delta.T, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("mode", ["flat", "pairwise", "hyperedge"])
def test_ties_singletons_and_empty_history_never_invent_directions(mode):
    keys, q, _, _ = arrays()
    net = HypergraphKeyCompiler(8, 16, mode)
    y = torch.tensor([[1., 1., 1., 1.], [0., 0., 0., 0.],
                      [1., float("nan"), float("nan"), float("nan")]])
    for query, outcome in ((q[:3], y), (q[:0], y[:0])):
        delta, bias = net(keys, query, outcome, torch.isfinite(outcome))
        assert delta.eq(0).all() and bias.eq(0).all()


def test_joint_edge_receives_gradients_and_deploys_as_single_index():
    keys, q, y, mask = arrays()
    net = HypergraphKeyCompiler(8, 16, "hyperedge")
    delta, bias = net(keys, q[:2], y[:2], mask[:2])
    score = q[:2] @ (keys + delta).T + bias
    loss, _ = expected_utility_loss(score, y[:2], mask[:2])
    loss.backward()
    assert net.edge_read.in_proj_weight.grad.abs().sum() > 0
    index = torch.cat((keys + delta, bias[:, None]), -1)
    torch.testing.assert_close(torch.cat((q[:2], torch.ones(2, 1)), -1) @ index.T, score)


def test_member_outcome_association_survives_old_summary_collision():
    keys, q, _, _ = arrays()
    a, b = torch.tensor([[1., 1., 0., 0.]]), torch.tensor([[1., 0., 1., 0.]])
    mask = torch.ones_like(a, dtype=torch.bool)
    ea, da, _ = observed_evidence_bank(q[:1], keys, a, mask)
    eb, db, _ = observed_evidence_bank(q[:1], keys, b, mask)
    # Same focal outcome and mean peer outcome/key hide which competitor won.
    torch.testing.assert_close(ea[0], eb[0])
    torch.testing.assert_close(da[0], db[0])
    for mode in ('pairwise', 'hyperedge'):
        net = HypergraphKeyCompiler(8, 16, mode).eval()
        ma = net.relation_messages(keys, q[:1], a, mask)[0]
        mb = net.relation_messages(keys, q[:1], b, mask)[0]
        assert (ma - mb).abs().max() > 1e-5
    # This establishes information retention; it does not imply hyperedge wins.


def test_graph_adapter_keeps_singleton_outcomes_and_version_group_permissions():
    refs = (SkillRef("a", "a" * 64), SkillRef("b", "b" * 64), SkillRef("c", "c" * 64))
    graph = SkillEvidenceGraph()
    graph.append(EpisodeEvidence("e", "task", "db", "0" * 64, "train", "executor",
        "environment", 17, "1" * 64, (ObservedArm((refs[0],), 1, "trace0"),
                                      ObservedArm((refs[1], refs[2]), 1, "trace1"))))
    graph.freeze()
    args = dict(task_ids=["task"], queries={"task": torch.ones(8)}, skill_refs=refs,
                excluded_groups=[], executor_id="executor", environment_id="environment", seed=17)
    result = graph_execution_matrix(graph, **args)
    assert result["observed"].tolist() == [[True, False, False]]
    assert torch.isnan(result["outcomes"][0, 1:]).all()
    with pytest.raises(ValueError, match="lacks authorized"):
        graph_execution_matrix(graph, **{**args, "excluded_groups": ["db"]})
    with pytest.raises(ValueError, match="lacks authorized"):
        graph_execution_matrix(graph, **{**args, "executor_id": "another_executor"})
