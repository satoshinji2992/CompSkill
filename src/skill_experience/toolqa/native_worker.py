"""Official ToolQA tools in a credential-free, disposable Docker container."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys


def main():
    # Libraries may print during initialization; stdout is reserved for protocol.
    with contextlib.redirect_stdout(sys.stderr):
        import sragents.toolqa as official
        from sragents.toolqa import ToolEnvironment
        env = ToolEnvironment(Path('/corpus'))
    source = Path(official.__file__).parent
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(source.rglob('*.py'))}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    print(json.dumps({'ready': True, 'source_sha256': digest}), flush=True)
    try:
        for line in sys.stdin:
            value = json.loads(line)
            if set(value) != {'action'} or not isinstance(value['action'], str):
                raise ValueError('Only an action string is accepted')
            with contextlib.redirect_stdout(sys.stderr):
                observation = env.execute(value['action'])
            print(json.dumps({'observation': observation}), flush=True)
    finally:
        if getattr(env, '_sql_conn', None) is not None:
            env._sql_conn.close()


if __name__ == '__main__': main()

