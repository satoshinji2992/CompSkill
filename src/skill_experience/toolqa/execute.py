"""Sequential official ToolQA execution and separately gated official scoring."""
import argparse
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from skill_experience.io import read, write, sha, seal, verify
from .original import run_official


def official(source, module, expected):
    source = Path(source).resolve()
    sys.path.insert(0, str(source / 'src'))
    value = importlib.import_module(module)
    path = Path(value.__file__).resolve()
    if not path.is_relative_to(source) or sha(path) != expected:
        raise ValueError('Upstream source identity differs')
    return value


class Native:
    def __init__(self, args, directory):
        self.name = 'compskill-' + uuid.uuid4().hex
        self.log = (directory / 'native.stderr.log').open('x')
        self.closed = False
        command = ['docker', 'run', '--rm', '-i', '--name', self.name,
            '--network=none', '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
            '--pids-limit=256', '--memory=84g', '--memory-swap=84g', '--cpus=2',
            '--user=10001:10001', '--tmpfs=/tmp:rw,nosuid,size=16g',
            '--mount', f'type=bind,src={Path(args.corpus).resolve()},dst=/corpus,readonly',
            '--mount', f'type=bind,src={Path(args.hf_cache).resolve()},dst=/hf,readonly',
            '-e', 'HF_HOME=/hf', '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
            '-e', 'PYTHONDONTWRITEBYTECODE=1', args.native_image]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=self.log, text=True)

    def ready(self, expected):
        if json.loads(self.process.stdout.readline()) != {'ready': True, 'source_sha256': expected}:
            raise RuntimeError('Native initialization failed')

    def execute(self, action):
        self.process.stdin.write(json.dumps({'action': action}) + '\n'); self.process.stdin.flush()
        response = json.loads(self.process.stdout.readline())
        return response['observation']

    def close(self):
        # Exact per-episode container identity, never a broad process kill.
        try:
            if self.process.poll() is None:
                try: self.process.stdin.close()
                except BrokenPipeError: pass
                try: self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    subprocess.run(['docker', 'rm', '-f', self.name], check=True, capture_output=True, timeout=30)
                    self.process.wait(timeout=10)
            check = subprocess.run(['docker', 'ps', '-aq', '--filter', 'name=^/' + self.name + '$'],
                                    check=True, capture_output=True, text=True, timeout=15)
            if check.stdout.strip():
                raise RuntimeError('Owned container remains')
            self.closed = True
        finally:
            self.log.close()
            if self.process.stdout: self.process.stdout.close()


def complete_gate(root):
    verify(root, 'prepare.seal.json'); verify(root, 'review.seal.json'); verify(root, 'generation.seal.json')
    expected = [t['task_id'] for t in read(root / 'tasks.json')]
    records = read(root / 'receipts.json')
    if [r['task_id'] for r in records] != expected or not all(r['native_released'] for r in records):
        raise ValueError('Incomplete generation or native cleanup')
    return records


def generate(args):
    from .transport import Client
    root = Path(args.run)
    verify(root, 'prepare.seal.json'); verify(root, 'review.seal.json')
    settings = read(root / 'settings.json')
    module = official(args.sra, 'sragents.infer.engines.react', settings['react_sha256'])
    tasks, selected = read(root / 'tasks.json'), read(root / 'frozen_guidance.json')
    if [t['task_id'] for t in tasks] != [t['task_id'] for t in selected]:
        raise ValueError('Guidance identity/order differs')
    out = root / 'generation'; out.mkdir(exist_ok=False)
    records = []
    def alarm(*_): raise TimeoutError('Whole episode exceeded 900 seconds')
    previous = signal.signal(signal.SIGALRM, alarm)
    try:
        with Client(settings, role='actor') as client:
            for i, (task, chosen) in enumerate(zip(tasks, selected)):
                directory = out / f'{i:04d}'; directory.mkdir()
                native = agent = None
                errors = []; began = time.monotonic()
                client.reset_budget(seconds=900, output_limit=24000)
                signal.setitimer(signal.ITIMER_REAL, 900)
                try:
                    native = Native(args, directory); native.ready(settings['native_sha256'])
                    def call(messages, model):
                        call_id = client.calls + 1
                        write(directory / f'prompt_{call_id:02d}.json', messages)
                        text = client(messages, model)
                        write(directory / f'response_{call_id:02d}.json', {'text': text})
                        return text
                    agent = run_official(module, task['question'], native, chosen['guidance'],
                        dict(model=settings['model'], max_steps=20, max_tokens=16384), call)
                except Exception as exc:
                    errors.append(type(exc).__name__)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    if native is not None:
                        try: native.close()
                        except Exception as exc: errors.append('cleanup:' + type(exc).__name__)
                receipt = dict(task_id=task['task_id'], skill_id=chosen['skill_id'], errors=errors,
                    scoring_eligible=not errors, raw_output=agent.model_scratchpad if agent is not None else '',
                    native_released=bool(native is not None and native.closed), usage=client.usage,
                    calls=client.calls, elapsed_seconds=time.monotonic() - began)
                write(directory / 'receipt.json', receipt); records.append(receipt)
                if errors:
                    raise RuntimeError('Episode stopped; receipt preserved, no automatic replay or continuation')
        write(root / 'receipts.json', records)
        seal(root, 'generation.seal.json', [root / 'receipts.json', *out.rglob('*.json')])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)


def score(args):
    root = Path(args.run)
    records = complete_gate(root)  # Before importing the scorer or opening any labels.
    settings = read(root / 'settings.json')
    evaluate = official(args.sra, 'sragents.evaluate.datasets.toolqa', settings['scorer_sha256']).evaluate
    scoring_path = Path(args.labels)
    if sha(scoring_path) != args.labels_sha256:
        raise ValueError('Official label file identity differs')
    data = read(scoring_path)
    gold = {r['instance_id']: r for r in data}
    if not set(r['task_id'] for r in records) <= gold.keys():
        raise ValueError('Missing scoring identity')
    out = root / 'evaluation'; out.mkdir(exist_ok=False)
    results = []
    for r in records:
        detail = evaluate(r['raw_output'], gold[r['task_id']]) if r['scoring_eligible'] else None
        results.append(dict(task_id=r['task_id'], reward=int(detail['correct']) if detail else None, detail=detail))
    write(out / 'scores.json', results)
    write(out / 'source.json', {'labels_sha256': args.labels_sha256, 'scorer_sha256': settings['scorer_sha256']})
    seal(out, 'results.seal.json', list(out.iterdir()))


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(required=True)
    q = sub.add_parser('generate'); q.add_argument('--run', required=True); q.add_argument('--sra', required=True)
    for name in ('corpus', 'hf-cache', 'native-image'): q.add_argument('--' + name, required=True)
    q.set_defaults(handler=generate)
    q = sub.add_parser('score'); q.add_argument('--run', required=True); q.add_argument('--sra', required=True)
    q.add_argument('--labels', required=True); q.add_argument('--labels-sha256', required=True); q.set_defaults(handler=score)
    args = p.parse_args(); args.handler(args)


if __name__ == '__main__': main()
