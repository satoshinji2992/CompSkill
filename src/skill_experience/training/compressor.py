"""Portable TRAIN-only driver for the released compressor recipe.

No training or downloads occur on import. The pure objective, masks, split rule,
and learning-rate schedule are the original functions in recipe.py.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

from skill_experience.config import RELEASE_ROOT
from skill_experience.io import read, write, sha, seal, verify
from .recipe import (SOURCE_HASHES, REVISION, field_text, normalized,
                     batch_masks, objective, lr_factor)

QUERY_PREFIX = 'Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: '


def training_rows(data):
    values = {}
    for name, expected in SOURCE_HASHES.items():
        path = Path(data) / 'data' / name / 'train.jsonl'
        if sha(path) != expected:
            raise ValueError('Official TRAIN bytes differ: ' + name)
        values[name] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    skills, queries, qrels = [values[n] for n in ('skills', 'queries', 'qrels')]
    if tuple(map(len, (skills, queries, qrels))) != (10123, 63259, 127190):
        raise ValueError('TRAIN counts differ')
    positives = defaultdict(set)
    for row in qrels:
        if row['relevance'] != 1:
            raise ValueError('Unexpected relevance value')
        positives[row['query_id']].add(row['skill_id'])
    if any(positives[q['id']] != set(q['skill_ids']) for q in queries):
        raise ValueError('Query labels disagree with TRAIN qrels')
    return skills, queries


def fetch(args):
    from huggingface_hub import snapshot_download
    snapshot_download('ThakiCloud/SKILLRET', repo_type='dataset', revision=REVISION,
                      local_dir=args.output, allow_patterns=['data/*/train.jsonl', 'README.md'])
    training_rows(args.output)


def encode(args):
    import torch
    from safetensors.torch import save_file
    from sentence_transformers import SentenceTransformer
    skills, queries = training_rows(args.data)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError('BF16 CUDA encoding required')
    model = SentenceTransformer(args.encoder, device='cuda', local_files_only=True,
                                model_kwargs={'torch_dtype': torch.bfloat16})
    model.max_seq_length = 8192
    data = {}
    for name, texts in [('skills', [field_text(s) for s in skills]),
                        ('queries', [QUERY_PREFIX + q['query'] for q in queries])]:
        data[name] = model.encode(texts, batch_size=args.batch_size,
                                  normalize_embeddings=True, convert_to_tensor=True,
                                  show_progress_bar=True).detach().float().cpu()
        if data[name].shape != (len(texts), 4096):
            raise ValueError('Expected full 4096-dimensional frozen embeddings')
    save_file(data, str(out / 'inputs.safetensors'))
    write(out / 'ids.json', {'skills': [s['id'] for s in skills],
                            'queries': [q['id'] for q in queries]})
    write(out / 'train.json', queries)
    # This is the actual TRAIN-only split used by the released checkpoint.
    write(out / 'split.json', read(RELEASE_ROOT / 'configs/compressor_split.json'))
    write(out / 'encoding.json', {'encoder': 'Qwen/Qwen3-Embedding-8B',
        'precision': 'BF16', 'dimension': 4096, 'maximum_tokens': 8192,
        'serialization': 'newline-separated nonempty name/description/skill_md',
        'dataset_revision': REVISION, 'query_prefix': QUERY_PREFIX,
        'note': 'Fresh cache; numerical identity with the original hardware/cache is not guaranteed.'})
    seal(out, 'inputs.seal.json', list(out.iterdir()))


def fit(args):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
    from skill_experience.retrieval.compressed_skill_key import CompressedSkillKey
    root, out = Path(args.inputs), Path(args.output)
    verify(root, 'inputs.seal.json')
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 8 * 2**30:
        raise RuntimeError('Training requires a CUDA device with at least 8 GiB free')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    split, ids, all_rows = [read(root / n) for n in ('split.json', 'ids.json', 'train.json')]
    allowed = set(split['fit'] if args.screen else split['fit'] + split['validation'] + split['mixed'])
    rows = [r for r in all_rows if r['id'] in allowed]
    if len(rows) != len(allowed):
        raise ValueError('Missing or duplicate training rows')
    dup = defaultdict(set)
    for r in rows:
        dup[normalized(r['query'])].update(r['skill_ids'])
    rows = [{**r, 'duplicate_skills': sorted(dup[normalized(r['query'])])} for r in rows]
    candidates = [s for s in ids['skills'] if not args.screen or s not in set(split['held_skills'])]
    si, qi = [{s: i for i, s in enumerate(ids[n])} for n in ('skills', 'queries')]
    data = load_file(str(root / 'inputs.safetensors'))
    cond = data['skills'][[si[s] for s in candidates]].cuda()
    native = F.normalize(cond[:, :512].float(), dim=-1)
    queries = F.normalize(data['queries'][[qi[r['id']] for r in rows], :512].float(), dim=-1).cuda()
    total = math.ceil(len(rows) / 256) * 20  # Final refit keeps the 20-epoch schedule prefix.
    epochs = 20 if args.screen else 5
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed + 317)
    model = CompressedSkillKey().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    write(out / 'recipe.json', {'seed': args.seed, 'epochs': epochs, 'schedule_epochs': 20,
        'batch_size': 256, 'lr': 1e-4, 'weight_decay': .01, 'temperature': .05,
        'anchor_weight': .1, 'clip_norm': 1., 'maximum_seconds': 3600,
        'screen': args.screen, 'inputs_sha256': sha(root / 'inputs.seal.json')})
    start, step = time.monotonic(), 0
    with (out / 'curve.jsonl').open('x') as log:
        for epoch in range(1, epochs + 1):
            sums = [0., 0., 0.]
            for batch in torch.randperm(len(rows), generator=rng).split(256):
                if time.monotonic() - start > 3600:
                    raise TimeoutError('Training time budget exhausted')
                for group in optimizer.param_groups:
                    group['lr'] = 1e-4 * lr_factor(step, total)
                positive, valid = batch_masks([rows[i] for i in batch.tolist()], candidates)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    keys = model(cond)
                losses = objective(keys, native, queries[batch], positive.cuda(), valid.cuda())
                if not torch.isfinite(losses[0]):
                    raise FloatingPointError('Nonfinite objective')
                losses[0].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                step += 1
                for j, value in enumerate(losses):
                    sums[j] += value.item() * len(batch)
            log.write(json.dumps({'epoch': epoch, 'step': step,
                                  'loss': [x / len(rows) for x in sums]}) + '\n')
            log.flush()
            if (args.screen and epoch in (1, 2, 5, 10, 20)) or epoch == epochs:
                name = f'epoch_{epoch}.safetensors' if args.screen else 'final.safetensors'
                save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(out / name))
    seal(out, 'training.seal.json', list(out.iterdir()))


def compress(args):
    import torch
    from safetensors.torch import load_file, save_file
    from skill_experience.retrieval.compressed_skill_key import CompressedSkillKey
    checkpoint = Path(args.checkpoint)
    if sha(checkpoint) != read(RELEASE_ROOT / 'models/config.json')['sha256']:
        raise ValueError('This inference entry requires the released primary checkpoint')
    model = CompressedSkillKey().eval()
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    raw = load_file(args.features)[args.tensor]
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    with torch.no_grad():
        keys = torch.cat([model(batch) for batch in raw.split(128)])
    save_file({'keys': keys}, args.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('fetch'); q.add_argument('--output', required=True); q.set_defaults(run=fetch)
    q = sub.add_parser('encode')
    for name in ('data', 'encoder', 'output'): q.add_argument('--' + name, required=True)
    q.add_argument('--batch-size', type=int, default=8); q.set_defaults(run=encode)
    q = sub.add_parser('fit')
    q.add_argument('--inputs', required=True); q.add_argument('--output', required=True)
    q.add_argument('--screen', action='store_true'); q.add_argument('--seed', type=int, choices=[17, 29, 43], default=17)
    q.set_defaults(run=fit)
    q = sub.add_parser('compress')
    q.add_argument('--features', required=True); q.add_argument('--output', required=True)
    q.add_argument('--tensor', default='skills')
    q.add_argument('--checkpoint', default=str(RELEASE_ROOT / 'models/compressor.safetensors')); q.set_defaults(run=compress)
    args = p.parse_args(); args.run(args)


if __name__ == '__main__': main()
