"""KeyOpt: support-calibrated execution feedback on top of a frozen reranker.

Score for task t and candidate skill s:

    S(t, s) = r_t(s) + beta_s + clip((q_t - mu)^T dk_s, -c, c) + alpha * c_t(s)

* beta_s   reliability offset, fitted on matched discordant pairs (pairwise logistic).
* dk_s     query-conditional residual, fitted on the same pairs on top of beta, with
           a keep penalty that holds the residual near zero on every support query.
* c        cap on the residual's score change.
* c_t(s)   hypergraph reading: similarity-weighted advantage of s on neighbouring
           support events (outcome minus the event's mean outcome).
* alpha, the keep strength, the cap c and whether the residual is used are chosen
  by five-fold source-group cross-validation on support data only, relative to the
  reranker's own score scale. Nothing here reads evaluation outcomes.

Inputs are plain Python/numpy objects; no benchmark I/O and no model calls.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
L2_GRID = (1e-1, 1e-2, 1e-3)
RADIUS_GRID = (0.3, 0.6, 1.0, 2.0, float('inf'))
CAP_GRID = (0.05, 0.1, 0.2, 0.3, 0.5, float('inf'))  # max |residual score change|
KEEP_GRID = (0.0, 1.0, 10.0)


def clip_rows(D, rho):
    if not np.isfinite(rho):
        return D
    n = np.linalg.norm(D, axis=1, keepdims=True)
    return D * np.minimum(1.0, rho / np.maximum(n, 1e-12))


@dataclass
class SupportData:
    """outcomes[task][skill] = mean observed success; unknown outcomes are simply absent."""
    outcomes: dict
    query: dict            # task -> unit 512-d query
    rerank: dict           # task -> {skill: reranker score}
    group: dict = field(default_factory=dict)   # task -> source group (default: task id)

    def grp(self, t):
        return self.group.get(t, t)


def discordant_pairs(outcomes):
    pairs = []
    for t, d in outcomes.items():
        items = sorted(d.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (a, ya), (b, yb) = items[i], items[j]
                if ya != yb:
                    w, l = (a, b) if ya > yb else (b, a)
                    pairs.append((t, w, l, abs(ya - yb)))
    return pairs


def _fit(pairs, data, skills, *, residual, l2, steps=400, tau=0.1, lr=0.03, seed=17, beta=None, keep=0.0):
    import torch
    torch.manual_seed(seed)
    idx = {s: i for i, s in enumerate(skills)}
    n = len(skills)
    dim = len(next(iter(data.query.values())))
    if not pairs:
        return np.zeros(n), np.zeros((n, dim)), np.zeros(dim)
    Q = torch.tensor(np.stack([data.query[p[0]] for p in pairs]), dtype=torch.float32)
    mu = Q.mean(0)
    wi = torch.tensor([idx[p[1]] for p in pairs]); li = torch.tensor([idx[p[2]] for p in pairs])
    wt = torch.tensor([p[3] for p in pairs], dtype=torch.float32)
    base = torch.tensor([data.rerank[p[0]][p[1]] - data.rerank[p[0]][p[2]] for p in pairs], dtype=torch.float32)
    b = torch.zeros(n, requires_grad=beta is None)
    if beta is not None:
        b = torch.tensor(beta, dtype=torch.float32)
    D = torch.zeros(n, dim, requires_grad=residual)
    params = [p for p in (b, D) if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    Qall = torch.tensor(np.stack(list(data.query.values())), dtype=torch.float32) - mu
    for _ in range(steps):
        opt.zero_grad()
        c = lambda ix: b[ix] + ((Q - mu) * D[ix]).sum(-1)
        z = (base + c(wi) - c(li)) / tau
        loss = (torch.nn.functional.softplus(-z) * wt).mean() + l2 * ((b ** 2).mean() + (D ** 2).sum(1).mean())
        if residual and keep:
            # Keep the conditional term near zero on every support query; deviate only where pairs demand it.
            loss = loss + keep * (Qall @ D.T).pow(2).mean()
        loss.backward(); opt.step()
    return b.detach().numpy(), D.detach().numpy(), mu.numpy()


def reading(outcomes, data, t, s, *, M=16, gamma=0.3, T=0.1, exclude_group=None):
    """Similarity-weighted advantage of s on admitted neighbouring support events."""
    rows = []
    q = data.query[t]
    for u, d in outcomes.items():
        if u == t or s not in d or len(d) < 2 or len(set(d.values())) == 1:
            continue
        if exclude_group is not None and data.grp(u) == exclude_group:
            continue
        sim = float(q @ data.query[u])
        if sim > gamma:
            rows.append((sim, d[s] - float(np.mean(list(d.values())))))
    if not rows:
        return 0.0
    rows.sort(key=lambda r: -r[0]); rows = rows[:M]
    w = np.exp(np.array([r[0] for r in rows]) / T); w /= w.sum()
    return float((w * np.array([r[1] for r in rows])).sum())


@dataclass
class KeyOpt:
    skills: dict
    beta: np.ndarray
    delta: np.ndarray
    mu: np.ndarray
    alpha: float
    use_residual: bool
    events: dict
    data: SupportData
    cv: dict
    radius: float = float('inf')

    def correction(self, t, s, query):
        i = self.skills[s]
        c = float(self.beta[i])
        if self.use_residual:
            c += float(np.clip((query[t] - self.mu) @ self.delta[i], -self.radius, self.radius))
        if self.alpha:
            c += self.alpha * reading(self.events, _with_query(self.data, query), t, s)
        return c

    def score(self, t, candidates, rerank, query):
        return {s: float(rerank[t][s]) + self.correction(t, s, query) for s in candidates}


def _with_query(data, query):
    q = dict(data.query); q.update(query)
    return SupportData(data.outcomes, q, data.rerank, data.group)


def _select_rate(outcomes, data, scorer):
    num = den = 0
    for t, d in outcomes.items():
        cands = [s for s in d if s in data.rerank.get(t, {})]
        if len(cands) < 2 or len({d[s] for s in cands}) == 1:
            continue
        sc = {s: data.rerank[t][s] + scorer(t, s) for s in cands}
        num += d[max(cands, key=lambda s: sc[s])]; den += 1
    return num / den if den else float('nan'), den


def fit(data: SupportData, skills, *, folds=5, alpha_grid=ALPHA_GRID, l2_grid=(1e-2,), radius_grid=CAP_GRID, keep_grid=KEEP_GRID):
    """Choose (alpha, keep, residual, cap) by source-group CV on support; refit on all support."""
    groups = sorted({data.grp(t) for t in data.outcomes})
    fold = {g: int(hashlib.sha256(str(g).encode()).hexdigest(), 16) % folds for g in groups}
    configs = [(a, kp, r, rho) for a in alpha_grid for kp in keep_grid for r in (False, True)
               for rho in (radius_grid if r else (0.0,)) if r or kp == keep_grid[0]]
    idx = {s: i for i, s in enumerate(skills)}
    stats = defaultdict(lambda: [0.0, 0])
    for f in range(folds):
        tr = {t: d for t, d in data.outcomes.items() if fold[data.grp(t)] != f}
        te = {t: d for t, d in data.outcomes.items() if fold[data.grp(t)] == f}
        pairs = discordant_pairs(tr)
        b0, _, _ = _fit(pairs, data, skills, residual=False, l2=l2_grid[0])
        res = {kp: _fit(pairs, data, skills, residual=True, l2=l2_grid[0], beta=b0, keep=kp) for kp in keep_grid}
        for a, kp, r, rho in configs:
            _, D, mu = res[kp]
            def sc(t, s, a=a, r=r, D=D, mu=mu, rho=rho):
                c = b0[idx[s]] + (float(np.clip((data.query[t] - mu) @ D[idx[s]], -rho, rho)) if r else 0.0)
                return c + (a * reading(tr, data, t, s, exclude_group=data.grp(t)) if a else 0.0)
            rate, n = _select_rate(te, data, sc)
            if n:
                stats[(a, kp, r, rho)][0] += rate * n; stats[(a, kp, r, rho)][1] += n
    cv = {c: v[0] / v[1] for c, v in stats.items() if v[1]}
    # Prefer the higher CV rate; ties go to smaller alpha, no residual, smaller cap, stronger keep.
    best = (max(cv, key=lambda c: (round(cv[c], 6), -c[0], not c[2], -c[3], c[1])) if cv
            else (0.0, l2_grid[0], False, 0.0))  # no contrast anywhere: leave the reranker unchanged
    a, kp, r, rho = best
    pairs = discordant_pairs(data.outcomes)
    beta, _, _ = _fit(pairs, data, skills, residual=False, l2=l2_grid[0])
    _, delta, mu = _fit(pairs, data, skills, residual=True, l2=l2_grid[0], beta=beta, keep=kp) if r else (None, np.zeros((len(skills), len(mu_of(data)))), mu_of(data))
    return KeyOpt(skills={s: i for i, s in enumerate(skills)}, beta=beta, delta=delta, mu=mu, alpha=a,
                    use_residual=r, events=data.outcomes, data=data,
                    cv={f'alpha={c[0]},keep={c[1]},residual={c[2]},cap={c[3]}': v for c, v in sorted(cv.items())},
                    radius=rho)


def mu_of(data):
    return np.mean(np.stack(list(data.query.values())), 0)
