"""Assemble portable inputs from public text and separately frozen support artifacts.

The public-task input must already be answer-free. This command never accepts a
scoring file, never collects support feedback, and never fits a new model.
"""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
from skill_experience.config import RELEASE_ROOT
from skill_experience.io import read, write, text_sha, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus', 'public-tasks', 'features', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--executor', choices=('gpt', 'deepseek'),
                   help='Use the released support, calibrations and settings for this executor')
    for name in ('support', 'keyopt', 'fitted', 'settings'):
        p.add_argument('--' + name)
    args = p.parse_args()
    supplied = (args.support, args.keyopt, args.fitted, args.settings)
    if args.executor:
        if any(supplied):
            p.error('--executor cannot be combined with custom support, calibration or settings')
        data = RELEASE_ROOT / 'data/toolqa' / args.executor
        manifest = read(data.parent / 'provenance.json')['models'][args.executor]['files']
        for name, field in [('support.json', 'support'), ('keyopt.safetensors', 'keyopt'),
                            ('fitted.safetensors', 'fitted'), ('settings.json', 'settings')]:
            if sha(data / name) != manifest[name]['sha256']:
                raise ValueError('Released input hash mismatch: ' + name)
            setattr(args, field, str(data / name))
    elif not (args.support and args.settings and (args.keyopt or args.fitted)):
        p.error('Provide --executor, or --support, --settings and --keyopt and/or --fitted')
    rows = read(args.public_tasks)
    if not isinstance(rows, list) or any(set(r) != {'task_id', 'source_group', 'question'} for r in rows):
        raise ValueError('Only answer-free public rows are accepted')
    tasks = {t['task_id']: t for t in rows}
    if len(tasks) != len(rows): raise ValueError('Duplicate task')
    with zipfile.ZipFile(args.corpus) as z:
        corpus = {s['skill_id']: s for s in json.loads(z.read('corpus.json'))}
    skills = []
    for entry in read(RELEASE_ROOT / 'configs/toolqa_bank.json'):
        s = corpus[entry['skill_id']]
        if text_sha(s['content']) != entry['document_sha256']:
            raise ValueError('Public corpus document differs')
        skills.append(dict(entry, name=s['name'], description=s['description'],
                           body=s['content'], document=s['content']))
    ordered = []
    for t in read(RELEASE_ROOT / 'configs/toolqa_split.json')['future']:
        value = tasks[t['task_id']]
        if value['source_group'] != t['source_group']: raise ValueError('Group mismatch')
        ordered.append(value)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    write(out / 'tasks.json', ordered); write(out / 'skills.json', skills)
    for src, dest in [(args.support, 'support.json'), (args.features, 'features.safetensors'),
                      (args.keyopt, 'keyopt.safetensors'), (args.fitted, 'fitted.safetensors'),
                      (args.settings, 'settings.json')]:
        if src:
            shutil.copyfile(src, out / dest)


if __name__ == '__main__': main()
