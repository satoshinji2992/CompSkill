"""Refit the final KeyOpt for one executor from the released support data only.

Uses the executor's support ledger (known outcomes only), its frozen 512-d support
queries and the frozen reranker scores of the support arms. Target-task outcomes
are never read. Five-fold source-group cross-validation selects alpha, the keep
strength, the score cap and whether the residual is used, then refits on all support.
"""
import argparse
from pathlib import Path

import numpy as np

from skill_experience.config import RELEASE_ROOT
from skill_experience.io import read


def support_data(executor):
    from skill_experience.retrieval.keyopt import SupportData
    root = RELEASE_ROOT / 'data/toolqa'
    scores = read(root / 'support_reranker_scores.json')
    outcomes, query, rerank, group = {}, {}, {}, {}
    for e in read(root / executor / 'support.json'):
        known = {a['skill_id']: float(a['outcome']) for a in e['arms'] if a['outcome'] is not None}
        if known:
            t, q = e['task_id'], np.asarray(e['query'], dtype=np.float32)
            outcomes[t], query[t], rerank[t], group[t] = known, q / np.linalg.norm(q), scores[t], e['source_group']
    return SupportData(outcomes, query, rerank, group)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executor', choices=('gpt', 'deepseek'), required=True)
    p.add_argument('--output', required=True, help='New safetensors file')
    args = p.parse_args()
    from safetensors import safe_open
    from safetensors.numpy import save_file
    from skill_experience.retrieval.keyopt import fit
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    skills = [s['skill_id'] for s in read(RELEASE_ROOT / 'configs/toolqa_bank.json')]
    m = fit(support_data(args.executor), skills)
    meta = dict(alpha=str(m.alpha), score_cap=str(m.radius), use_residual=str(m.use_residual))
    tensors = dict(beta=m.beta, delta=m.delta, mu=m.mu)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.astype(np.float32) for k, v in tensors.items()}, str(out), metadata=meta)
    with safe_open(str(RELEASE_ROOT / 'data/toolqa' / args.executor / 'keyopt.safetensors'), 'np') as f:
        print(dict(meta, same_selection_as_released=all(f.metadata()[k] == v for k, v in meta.items()),
                   max_abs_difference={k: float(np.abs(f.get_tensor(k) - v).max()) for k, v in tensors.items()}))


if __name__ == '__main__':
    main()
