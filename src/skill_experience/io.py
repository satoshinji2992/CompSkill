"""Write-once artifacts and content-bound, relative-path manifests."""
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def seal(root, name, files):
    root = Path(root).resolve()
    records = {}
    for path in files:
        path = Path(path).resolve()
        records[str(path.relative_to(root))] = sha(path)
    write(root / name, {'files': records})


def verify(root, name):
    root = Path(root).resolve()
    records = read(root / name)['files']
    if not records:
        raise ValueError('Empty manifest')
    for rel, expected in records.items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError('Manifest mismatch: ' + rel)
    return records

