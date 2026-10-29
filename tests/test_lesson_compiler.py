import pytest
import torch

from skill_experience.retrieval.lesson_compiler import LessonKeyCompiler, lesson_evidence_bank
from skill_experience.retrieval.memory_compiler import support_mask


@pytest.mark.parametrize("mode", ["mean", "attention"])
def test_masked_future_evidence_and_empty_skill_do_not_change_keys(mode):
    torch.manual_seed(0)
    net = LessonKeyCompiler(dimension=8, lesson_dimension=4, hidden=16, mode=mode)
    torch.nn.init.normal_(net.head.weight)
    keys, evidence = torch.randn(3, 8), torch.randn(3, 4, 32)
    owners = torch.tensor([[0, 1, 2, 3], [0, 1, -1, -1], [-1, -1, -1, -1]])
    mask = support_mask(owners, [0, 1], [2, 3])
    code, bias = net(keys, evidence, mask)
    changed = evidence.clone()
    changed[~mask] = float("nan")
    other_code, other_bias = net(keys, changed, mask)
    torch.testing.assert_close(code, other_code)
    torch.testing.assert_close(bias, other_bias)
    assert code[2].eq(0).all() and bias[2] == 0
    assert (code.norm(dim=1) + bias.abs()).max() <= .100001


def test_support_target_overlap_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        support_mask(torch.tensor([[0, 1]]), [0, 1], [1])


def test_tied_execution_records_remain_available_but_keep_source_ids():
    q, k = torch.randn(2, 8), torch.randn(3, 8)
    pairs = torch.tensor([[0, 1], [1, 2]])
    y = torch.tensor([[True, True], [False, True]])
    lessons = torch.randn(2, 2, 4)
    evidence, owners = lesson_evidence_bank(q, k, pairs, y, ~y, lessons, lessons, torch.ones(2, 2))
    assert int((owners >= 0).sum()) == 4
    assert evidence.shape[-1] == 32
    assert owners[0, 0].item() == 0


@pytest.mark.parametrize("mode", ["mean", "attention"])
def test_compiler_can_learn_a_nonzero_bounded_update(mode):
    torch.manual_seed(4)
    net = LessonKeyCompiler(dimension=8, lesson_dimension=4, hidden=16, mode=mode)
    keys, evidence, valid = torch.randn(2, 8), torch.randn(2, 3, 32), torch.ones(2, 3, dtype=torch.bool)
    target = torch.full((2, 8), .02)
    opt = torch.optim.Adam(net.parameters(), lr=.003)
    initial = float((net(keys, evidence, valid)[0] - target).square().mean().detach())
    for _ in range(30):
        code, _ = net(keys, evidence, valid)
        loss = (code - target).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.detach()) < initial * .5
