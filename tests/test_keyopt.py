import numpy as np

from skill_experience.retrieval.keyopt import SupportData, fit, reading


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _data(n=60, mode='conditional', seed=0):
    rng = np.random.default_rng(seed)
    outcomes, query, rerank = {}, {}, {}
    for i in range(n):
        t = f't{i}'
        side = 1 if i % 2 else -1
        q = np.zeros(8); q[0] = side; q[1:] = 0.2 * rng.standard_normal(7)
        query[t] = _unit(q)
        rerank[t] = {'a': 0.6, 'b': 0.55, 'c': 0.1}
        if mode == 'none':
            outcomes[t] = {'a': 1.0, 'b': 1.0}
        elif mode == 'reliable':
            outcomes[t] = {'a': 0.0, 'b': 1.0}
        else:  # b wins when q[0] > 0, a wins otherwise
            outcomes[t] = {'a': float(side < 0), 'b': float(side > 0)}
    return SupportData(outcomes, query, rerank)


def test_no_signal_leaves_reranker_unchanged():
    m = fit(_data(mode='none'), ['a', 'b', 'c'])
    assert np.allclose(m.beta, 0) and not m.use_residual and m.alpha == 0.0


def test_uniform_advantage_is_an_offset():
    m = fit(_data(mode='reliable'), ['a', 'b', 'c'])
    assert m.beta[m.skills['b']] > m.beta[m.skills['a']]


def test_conditional_advantage_flips_choice_by_query():
    data = _data(mode='conditional')
    m = fit(data, ['a', 'b', 'c'])
    assert m.use_residual or m.alpha > 0
    pos, neg = {'x': _unit([1, 0, 0, 0, 0, 0, 0, 0])}, {'x': _unit([-1, 0, 0, 0, 0, 0, 0, 0])}
    rr = {'x': {'a': 0.6, 'b': 0.55, 'c': 0.1}}
    sp = m.score('x', ['a', 'b'], rr, pos); sn = m.score('x', ['a', 'b'], rr, neg)
    assert max(sp, key=sp.get) == 'b' and max(sn, key=sn.get) == 'a'


def test_reading_ignores_unknowns_and_excluded_group():
    data = _data(mode='conditional', n=10)
    data.outcomes['t1'] = {'a': 1.0}  # single known arm: no contrast, must be ignored
    data.group = {t: 'g' for t in data.outcomes}
    assert reading(data.outcomes, data, 't0', 'a', exclude_group='g') == 0.0
