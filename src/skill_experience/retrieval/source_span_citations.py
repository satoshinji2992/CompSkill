"""Deterministic, lossless document line citations; no semantic quote repair."""
from __future__ import annotations
import copy
import hashlib
import json


OLD_RULE = 'quote must be a short exact substring of the SELECTED skill\'s document (preserve punctuation).'
NEW_RULE = ('quote must be ONE bracketed source-line ID printed before a nonempty line of the SELECTED '
            'skill document, for example [doc2:L00003]. Return only that ID as quote, not a transcription. '
            'Line IDs are citation metadata; all original document characters are preserved. '
            'Never cite a line ID belonging to another skill.')


def sha_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def annotate(document, prefix):
    lines, spans, offset = [], {}, 0
    for i, line in enumerate(document.splitlines(keepends=True), 1):
        label = f'[{prefix}:L{i:05d}]'
        lines.append(label + ' ' + line)
        spans[label] = dict(start=offset, end=offset+len(line), text=line,
                            nonempty=bool(line.strip()), document_sha256=sha_text(document))
        offset += len(line)
    assert offset == len(document)
    return ''.join(lines), spans


def request(original):
    q = copy.deepcopy(original)
    if q['prompt'][0]['content'].count(OLD_RULE) != 1:
        raise ValueError('Exact original review instruction missing')
    q['prompt'][0]['content'] = q['prompt'][0]['content'].replace(OLD_RULE, NEW_RULE)
    payload = json.loads(q['prompt'][1]['content']); catalog = {}
    for i, candidate in enumerate(payload['candidates'], 1):
        document = candidate['document']; sid = candidate['skill_id']
        if candidate['document_sha256'] != sha_text(document):
            raise ValueError('Original document hash mismatch')
        shown, spans = annotate(document, f'doc{i}')
        candidate['document'] = shown
        for key, span in spans.items():
            catalog[key] = dict(span, skill_id=sid)
        recovered = ''.join(span['text'] for span in spans.values())
        if recovered != document:
            raise ValueError('Citation view lost document characters')
    q['prompt'][1]['content'] = json.dumps(payload, ensure_ascii=False)
    return q, catalog


def validate(text, prompt, catalog):
    payload = json.loads(prompt[-1]['content'])
    docs = {c['skill_id']:c for c in payload['candidates']}
    events = {e['event_id'] for e in payload['history']}
    raw_choice = None; error = None; span = None
    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError('object')
        raw_choice = value.get('skill_id')
        if set(value) != {'skill_id','quote','evidence_ids','reason'}:
            raise ValueError('schema')
        if not isinstance(raw_choice,str) or raw_choice not in docs:
            raise ValueError('skill identity')
        if not isinstance(value['quote'], str) or value['quote'] not in catalog:
            raise ValueError('source-line identity')
        span = catalog[value['quote']]
        if span['skill_id'] != raw_choice or not span['nonempty']:
            raise ValueError('source-line attribution')
        if span['document_sha256'] != docs[raw_choice]['document_sha256']:
            raise ValueError('source version')
        if not isinstance(value['evidence_ids'],list) or any(
                not isinstance(e,str) or e not in events for e in value['evidence_ids']):
            raise ValueError('evidence attribution')
        if not isinstance(value['reason'],str) or not value['reason'].strip():
            raise ValueError('reason')
        return dict(value, valid=True, raw_choice=raw_choice, source_span=span,
                    quote_text=span['text'], fallback_applied=False, learning_eligible=True,
                    validation_error=None)
    except (ValueError,KeyError,TypeError) as exc:
        error = str(exc)
    return dict(skill_id=payload['proposed_skill_id'], quote='', evidence_ids=[],
                reason='invalid output; proposed-key fallback',valid=False,raw_choice=raw_choice,
                source_span=None,quote_text='',fallback_applied=True,learning_eligible=False,
                validation_error=error)


def clean_response(text):
    text = (text or '').strip()
    if text.startswith('```'):
        text = text.strip('`'); text = text[text.find('{'):] if '{' in text else text
    return text
