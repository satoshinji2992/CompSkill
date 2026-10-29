"""Explicit OpenAI-compatible API adapter; no credentials are written to disk.

This portable adapter does not reproduce the original RPC transport used for historical GPT runs.
No automatic retries, health probes, model fallbacks, or tool calls.
"""
import os
import time


class Client:
    def __init__(self, settings, role):
        from openai import OpenAI
        if role not in ('review', 'actor') or settings['provider'] not in ('gpt', 'deepseek'):
            raise ValueError('Unsupported role/provider')
        self.settings, self.role = settings, role
        self.client = OpenAI(api_key=os.environ['COMPSKILL_API_KEY'],
                             base_url=os.environ['COMPSKILL_BASE_URL'], max_retries=0)
        self.reset_budget(seconds=300, output_limit=4000)

    def reset_budget(self, *, seconds, output_limit):
        self.deadline = time.monotonic() + seconds
        self.output_limit, self.output_used, self.calls = output_limit, 0, 0
        self.usage = []

    def __call__(self, messages, model):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0 or self.output_used >= self.output_limit:
            raise TimeoutError('Frozen time/output budget exhausted')
        if self.role == 'actor' and self.calls >= 20:
            raise RuntimeError('Actor call budget exhausted')
        if model != self.settings['model']:
            raise ValueError('Model drift')
        limit = self.settings['review_max_tokens'] if self.role == 'review' else 16384
        kwargs = dict(model=model, messages=messages, timeout=remaining)
        if self.settings['provider'] == 'gpt':
            kwargs.update(reasoning_effort='low', max_completion_tokens=limit)
        else:
            kwargs.update(max_tokens=limit, reasoning_effort='low',
                          extra_body={'thinking': {'type': 'enabled'}})
        self.calls += 1
        result = self.client.chat.completions.create(**kwargs)
        usage = result.usage.model_dump() if result.usage is not None else None
        self.usage.append(usage)
        # Unknown usage must not silently become zero or permit unbounded calls.
        if usage is None or usage.get('completion_tokens') is None:
            raise RuntimeError('Provider output usage missing; stopping with unknown usage')
        self.output_used += usage['completion_tokens']
        return result.choices[0].message.content or ''

    def close(self): self.client.close()
    def __enter__(self): return self
    def __exit__(self, *_): self.close()

