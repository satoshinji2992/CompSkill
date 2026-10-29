"""ToolQA Full: versioned support memory, bounded recall, and citation review.

This portable driver replaces research-machine orchestration, not the method
modules or upstream agent/evaluator. See docs/toolqa.md for input contracts.
"""
import argparse
import json
import math
from pathlib import Path
import shutil

from skill_experience.config import RELEASE_ROOT
from skill_experience.io import read, write, sha, text_sha, seal, verify
from .original import REVIEW_SYSTEM, history_events, guidance

ENVIRONMENT = 'toolqa-official-react-785'
REVIEW_MODES = ('exact_quote', 'source_line')
# keyopt: final KeyOpt with support-selected strengths (default).
# fixed_strength: the submitted configuration's frozen calibration plus memory correction.
CALIBRATIONS = {'keyopt': 'keyopt.safetensors', 'fixed_strength': 'fitted.safetensors'}


def review_mode(settings):
    mode = settings.get('review_mode', 'exact_quote')
    if mode not in REVIEW_MODES:
        raise ValueError('review_mode must be exact_quote or source_line')
    return mode


def calibration(settings):
    value = settings.get('calibration', 'keyopt')
    if value not in CALIBRATIONS:
        raise ValueError('calibration must be keyopt or fixed_strength')
    return value


def keyopt_scorer(path, skills, support):
    """Load a fitted KeyOpt; its conditional reading uses only known support outcomes."""
    import numpy as np
    from safetensors import safe_open
    from skill_experience.retrieval.keyopt import KeyOpt, SupportData
    with safe_open(str(path), 'np') as f:
        tensors, meta = {k: f.get_tensor(k) for k in f.keys()}, f.metadata()
    if {k: v.shape for k, v in tensors.items()} != {'beta': (len(skills),), 'delta': (len(skills), 512), 'mu': (512,)}:
        raise ValueError('Misaligned KeyOpt tensors')
    outcomes, groups, queries = {}, {}, {}
    for e in support:
        known = {a['skill_id']: float(a['outcome']) for a in e['arms'] if a['outcome'] is not None}
        if known:
            q = np.asarray(e['query'], dtype=np.float32)
            outcomes[e['task_id']], groups[e['task_id']], queries[e['task_id']] = known, e['source_group'], q / np.linalg.norm(q)
    return KeyOpt(skills={s['skill_id']: i for i, s in enumerate(skills)}, beta=tensors['beta'],
                  delta=tensors['delta'], mu=tensors['mu'], alpha=float(meta['alpha']),
                  use_residual=meta['use_residual'] == 'True', radius=float(meta['score_cap']),
                  events=outcomes, data=SupportData(outcomes, queries, {}, groups), cv={})


def make_review_request(original, mode):
    if mode == 'exact_quote':
        return dict(original, review_mode=mode)
    if mode == 'source_line':
        from skill_experience.retrieval.source_span_citations import request
        transformed, catalog = request(original)
        return dict(transformed, catalog=catalog, review_mode=mode)
    raise ValueError('Unsupported review mode')


def review_decision(response, req):
    from skill_experience.retrieval.source_span_citations import clean_response
    if req['review_mode'] == 'exact_quote':
        from skill_experience.retrieval.evidence_agent import parse
        decision = parse(clean_response(response), req['prompt'])
        return dict(decision, fallback_applied=not decision['valid'],
                    learning_eligible=decision['valid'])
    if req['review_mode'] == 'source_line':
        from skill_experience.retrieval.source_span_citations import validate
        return validate(clean_response(response), req['prompt'], req['catalog'])
    raise ValueError('Unsupported review mode')


def validate_public(tasks, skills):
    if not tasks or not skills:
        raise ValueError('Empty inputs')
    if len({t['task_id'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicate task')
    if len({s['skill_id'] for s in skills}) != len(skills):
        raise ValueError('Duplicate skill')
    for t in tasks:
        if set(t) != {'task_id', 'source_group', 'question'} or not all(isinstance(v, str) and v for v in t.values()):
            raise ValueError('Only public task identity, group, and question are permitted')
    for s in skills:
        if set(s) != {'skill_id', 'name', 'description', 'body', 'document', 'document_sha256'}:
            raise ValueError('Unexpected skill fields')
        if text_sha(s['document']) != s['document_sha256']:
            raise ValueError('Skill document changed')


def build_memory(keys, skills, support, executor_id):
    import torch
    from skill_experience.retrieval.task_key_memory import TaskKeyOptMemory
    from skill_experience.retrieval.skill_evidence_graph import SkillRef, EpisodeEvidence, ObservedArm
    refs = [SkillRef(s['skill_id'], s['document_sha256']) for s in skills]
    by_id = {r.skill_id: r for r in refs}
    memory = TaskKeyOptMemory(keys, refs, executor_id=executor_id, environment_id=ENVIRONMENT)
    events = []
    if len({e['task_id'] for e in support}) != len(support):
        raise ValueError('Duplicate support task')
    for e in sorted(support, key=lambda e: e['task_id']):
        if e['split'] != 'train' or e['executor_id'] != executor_id or e['environment_id'] != ENVIRONMENT:
            raise ValueError('Support must come from the same executor/environment and TRAIN split')
        q = torch.tensor(e['query'], dtype=torch.float32)
        if not torch.isfinite(q).all() or q.norm() == 0:
            raise ValueError('Support query must be finite and nonzero')
        q = q / q.norm()
        arms = e['arms']
        if len(arms) != 2 or len({a['skill_id'] for a in arms}) != 2:
            raise ValueError('Two distinct frozen alternative arms required')
        for a in arms:
            if a['document_sha256'] != by_id[a['skill_id']].document_sha256:
                raise ValueError('Support skill version differs')
            if a['outcome'] not in (0, 1, None):
                raise ValueError('Support outcome must be binary or unknown')
        eid = 'support:' + e['task_id']
        context = dict(evidence_id=eid, task_id=e['task_id'], source_group=e['source_group'],
                       query_sha256=text_sha(e['question']), context_sha256=e.get('context_sha256', text_sha(e['question'])), seed=0)
        slate = memory.retrieve(q, **context, top_k=2, candidate_refs=tuple(by_id[a['skill_id']] for a in arms),
                                selection_sha256=e['selection_sha256'])
        observed = tuple(ObservedArm((by_id[a['skill_id']],), a['outcome'], a['physical_id'])
                         for a in arms if a['outcome'] is not None)
        if not observed: continue
        memory.record(slate, EpisodeEvidence(**context, split='train', executor_id=executor_id,
                                             environment_id=ENVIRONMENT, arms=observed))
        if len(observed) == 2:
            events.append(dict(event_id=eid, task_id=e['task_id'], question=e['question'], query=e['query'],
                versions={a['skill_id']: a['document_sha256'] for a in arms},
                outcomes={a['skill_id']: a['outcome'] for a in arms},
                executor=e.get('history_executor_label', e['executor_label']), environment='official ToolQA ReAct'))
    return memory, events


def prepare(args):
    import torch
    from safetensors.torch import load_file
    from skill_experience.retrieval.evidence_candidate_router import EvidenceCandidateRouter
    source, out = Path(args.inputs), Path(args.output)
    tasks, skills, support, settings = [read(source / n) for n in ('tasks.json', 'skills.json', 'support.json', 'settings.json')]
    validate_public(tasks, skills)
    review_mode(settings)
    fitted = CALIBRATIONS[calibration(settings)]
    if set(t['task_id'] for t in tasks) & set(t['task_id'] for t in support):
        raise ValueError('Target task cannot be admitted as support')
    frozen_bank = read(RELEASE_ROOT / 'configs/toolqa_bank.json')
    split = read(RELEASE_ROOT / 'configs/toolqa_split.json')
    frozen_tasks = split['future']
    if {(t['task_id'], t['source_group']) for t in support} != {(t['task_id'], t['source_group']) for t in split['support']}:
        raise ValueError('Expected the frozen 318-task support scope')
    if settings.get('provider') not in ('gpt', 'deepseek'):
        raise ValueError('Unsupported provider')
    expected_review = 4000 if settings['provider'] == 'gpt' else 8000
    if settings.get('review_max_tokens') != expected_review:
        raise ValueError('Review output budget differs')
    if [{'skill_id': s['skill_id'], 'document_sha256': s['document_sha256']} for s in skills] != frozen_bank:
        raise ValueError('Expected the frozen 785-document ToolQA bank, in order')
    if [{'task_id': t['task_id'], 'source_group': t['source_group']} for t in tasks] != frozen_tasks:
        raise ValueError('Expected the frozen 332-question public split, in order')
    features = load_file(str(source / 'features.safetensors'))
    keys, queries = features['keys'].float(), features['queries'].float()
    if keys.shape != (len(skills), 512) or queries.shape != (len(tasks), 512):
        raise ValueError('Misaligned frozen features')
    if not torch.isfinite(keys).all() or not torch.isfinite(queries).all():
        raise ValueError('Nonfinite features')
    if not torch.allclose(queries.norm(dim=1), torch.ones(len(tasks)), atol=1e-5):
        raise ValueError('Unit queries required')
    memory, events = build_memory(keys, skills, support, settings['executor_id'])
    router = EvidenceCandidateRouter(memory, candidate_budget=10, protected_semantic=6, max_expansion=4)
    routes = []
    for t, q in zip(tasks, queries):
        route_query = q / q.norm()
        d = router.route(route_query, keys @ route_query, current_task_id=t['task_id'], current_source_group=t['source_group'],
                         allowed_history_task_ids=[e['task_id'] for e in support])
        routes.append(dict(task_id=t['task_id'], pool=[r.skill_id for r in d.candidates],
            semantic_pool=[r.skill_id for r in d.semantic_candidates],
            correction={r.skill_id: c for r, c in zip(d.ranked, d.corrections)},
            evidence_ids=list(d.evidence_ids), memory_sha256=d.memory_sha256))
    out.mkdir(parents=True, exist_ok=False)
    for name in ('tasks.json', 'skills.json', 'support.json', 'settings.json', 'features.safetensors', fitted):
        shutil.copyfile(source / name, out / name)
    write(out / 'routes.json', routes); write(out / 'events.json', events)
    seal(out, 'prepare.seal.json', list(out.iterdir()))


def rerank(args):
    import importlib
    import sys
    import torch
    root = Path(args.run); verify(root, 'prepare.seal.json')
    sys.path.insert(0, str(Path(args.skillrouter).resolve()))
    common, official = [importlib.import_module(n) for n in ('src.common', 'src.run_open_model_eval')]
    for module in (common, official):
        if not Path(module.__file__).resolve().is_relative_to(Path(args.skillrouter).resolve()):
            raise ValueError('Wrong SkillRouter import')
    model, tokenizer = common.load_reranker_model(args.model)
    model.to('cuda').eval()
    skills = {s['skill_id']: s for s in read(root / 'skills.json')}
    tasks = {t['task_id']: t for t in read(root / 'tasks.json')}
    out = root / 'reranking'; out.mkdir(exist_ok=False)
    scores = {}
    for route in read(root / 'routes.json'):
        ids = list(dict.fromkeys(route['semantic_pool'] + route['pool']))
        candidates = [dict(skills[s], name=skills[s]['name']) for s in ids]
        values = official.score_candidates_with_reranker(model, tokenizer, tasks[route['task_id']]['question'],
                                                         candidates, 'flat-full', 4096, 2, torch.device('cuda'))
        scores[route['task_id']] = {s: 1 / (1 + math.exp(-float(v))) for s, v in zip(ids, values)}
        if not all(math.isfinite(v) for v in scores[route['task_id']].values()):
            raise ValueError('Nonfinite reranker score')
    write(out / 'scores.json', scores)
    seal(root, 'rerank.seal.json', [out / 'scores.json'])


def select(args):
    from safetensors.torch import load_file
    from skill_experience.retrieval.reranker_calibration import score_adjustment
    root = Path(args.run)
    verify(root, 'prepare.seal.json'); verify(root, 'rerank.seal.json')
    skills = read(root / 'skills.json'); pos = {s['skill_id']: i for i, s in enumerate(skills)}
    tasks, routes, events = [read(root / n) for n in ('tasks.json', 'routes.json', 'events.json')]
    scores = read(root / 'reranking/scores.json'); settings = read(root / 'settings.json')
    q = load_file(str(root / 'features.safetensors'))['queries'].float()
    if calibration(settings) == 'keyopt':
        scorer = keyopt_scorer(root / 'keyopt.safetensors', skills, read(root / 'support.json'))
        queries = {t['task_id']: q[i].numpy() for i, t in enumerate(tasks)}
    else:
        adj = score_adjustment(q, load_file(str(root / 'fitted.safetensors')))
    requests = []
    for i, (task, route) in enumerate(zip(tasks, routes)):
        rank = scores[task['task_id']]
        if calibration(settings) == 'keyopt':
            # The final KeyOpt replaces both the fixed calibration and the memory correction.
            values = {s: rank[s] + scorer.correction(task['task_id'], s, queries) for s in route['pool']}
        else:
            values = {s: rank[s] + float(adj[i, pos[s]]) + route['correction'][s] for s in route['pool']}
        top = sorted(values, key=lambda s: (-values[s], pos[s]))[:8]
        history = history_events(events, top, q[i].tolist(), route['evidence_ids'])
        # Historical semantic baseline resolves equal reranker scores by pool order.
        baseline = max(route['semantic_pool'], key=lambda s: rank[s])
        payload = dict(question=task['question'], baseline_skill_id=baseline, proposed_skill_id=top[0],
            candidates=[dict(skill_id=s, document_sha256=skills[pos[s]]['document_sha256'],
                             document=skills[pos[s]]['document']) for s in top], history=history)
        system = REVIEW_SYSTEM.replace('GPT-5.5 (low reasoning)', settings['executor_label'])
        original = dict(task_id=task['task_id'], proposal=top[0], prompt=[dict(role='system', content=system),
                         dict(role='user', content=json.dumps(payload, ensure_ascii=False))])
        requests.append(make_review_request(original, review_mode(settings)))
    write(root / 'review_requests.json', requests)
    seal(root, 'selection.seal.json', [root / 'review_requests.json'])


def review(args):
    from .transport import Client
    root = Path(args.run)
    verify(root, 'prepare.seal.json'); verify(root, 'selection.seal.json')
    settings = read(root / 'settings.json')
    requests = read(root / 'review_requests.json')
    if any(req['review_mode'] != review_mode(settings) for req in requests):
        raise ValueError('Frozen review mode differs from settings')
    out = root / 'review'; out.mkdir(exist_ok=False)
    bank = {s['skill_id']: s for s in read(root / 'skills.json')}
    frozen = []
    with Client(settings, role='review') as client:
        for i, req in enumerate(requests):
            client.reset_budget(seconds=300, output_limit=settings['review_max_tokens'])
            # Transport failures stop this release run; they never become fabricated responses.
            response = client(req['prompt'], settings['model'])
            decision = review_decision(response, req)
            write(out / f'{i:04d}.json', dict(task_id=req['task_id'], response=response, decision=decision,
                                          usage=client.usage))
            frozen.append(dict(task_id=req['task_id'], skill_id=decision['skill_id'],
                               guidance=guidance(bank, [decision['skill_id']])))
    write(root / 'frozen_guidance.json', frozen)
    seal(root, 'review.seal.json', [root / 'frozen_guidance.json', *out.glob('*.json')])


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(required=True)
    q = sub.add_parser('prepare'); q.add_argument('--inputs', required=True); q.add_argument('--output', required=True); q.set_defaults(handler=prepare)
    for name, fn in [('rerank', rerank), ('select', select), ('review', review)]:
        q = sub.add_parser(name); q.add_argument('--run', required=True); q.set_defaults(handler=fn)
        if name == 'rerank':
            q.add_argument('--skillrouter', required=True); q.add_argument('--model', required=True)
    args = p.parse_args(); args.handler(args)


if __name__ == '__main__': main()
