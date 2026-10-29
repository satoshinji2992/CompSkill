"""Published-input integrity and evidence-review profile boundaries."""
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from skill_experience.config import RELEASE_ROOT
from skill_experience.io import read, sha, seal, write, text_sha
from skill_experience.toolqa.full import make_review_request, review_decision, review_mode, review
from skill_experience.toolqa.original import REVIEW_SYSTEM


def example_request():
    payload = dict(question='Public question', baseline_skill_id='a', proposed_skill_id='a',
        candidates=[dict(skill_id='a', document='Alpha text', document_sha256=text_sha('Alpha text')),
                    dict(skill_id='b', document='Beta text', document_sha256=text_sha('Beta text'))],
        history=[dict(event_id='known-event')])
    return dict(task_id='q', proposal='a', prompt=[dict(role='system',content=REVIEW_SYSTEM),
        dict(role='user',content=json.dumps(payload))])


def test_default_exact_quote_preserves_original_prompt_bytes():
    original = example_request()
    req = make_review_request(original, review_mode({}))
    assert req['review_mode'] == 'exact_quote'
    assert req['prompt'] == original['prompt']
    response = dict(skill_id='b', quote='Beta', evidence_ids=['known-event'], reason='applicable')
    result = review_decision(json.dumps(response), req)
    assert result['valid'] and result['skill_id'] == 'b'
    assert not result['fallback_applied']
    assert read(RELEASE_ROOT/'configs/toolqa_full.json')['review_mode'] == 'exact_quote'
    with pytest.raises(ValueError): review_mode({'review_mode':'unknown-mode'})


@pytest.mark.parametrize('quote,evidence', [('Alpha', []), ('[doc2:L00001]', []), ('Beta', ['unseen-event']), ('', [])])
def test_exact_quote_rejects_wrong_document_line_id_and_unsupplied_history(quote,evidence):
    req = make_review_request(example_request(), 'exact_quote')
    result = review_decision(json.dumps(dict(skill_id='b',quote=quote,evidence_ids=evidence,reason='x')),req)
    assert not result['valid'] and result['fallback_applied']
    assert result['skill_id'] == 'a' and not result['learning_eligible']


def test_frozen_mode_mismatch_stops_before_provider_or_output(tmp_path,monkeypatch):
    import skill_experience.toolqa.transport as transport
    write(tmp_path/'settings.json', {'review_mode':'exact_quote'})
    write(tmp_path/'review_requests.json', [make_review_request(example_request(),'source_line')])
    seal(tmp_path,'prepare.seal.json',[tmp_path/'settings.json'])
    seal(tmp_path,'selection.seal.json',[tmp_path/'review_requests.json'])
    def forbidden(*a,**k): pytest.fail('Provider constructed before validating frozen mode')
    monkeypatch.setattr(transport,'Client',forbidden)
    with pytest.raises(ValueError,match='mode differs'): review(SimpleNamespace(run=str(tmp_path)))
    assert not (tmp_path/'review').exists()


@pytest.mark.parametrize('executor,unknown', [('gpt',3),('deepseek',10)])
def test_published_support_bundle_preserves_scope_unknowns_and_only_public_fields(executor,unknown):
    root=RELEASE_ROOT/'data/toolqa'; bundle=root/executor
    manifest=read(root/'provenance.json')['models'][executor]
    for filename, entry in manifest['files'].items():
        assert sha(bundle/filename) == entry['sha256']
        assert (bundle/filename).stat().st_size == entry['bytes']
    settings=read(bundle/'settings.json'); records=read(bundle/'support.json')
    split=read(RELEASE_ROOT/'configs/toolqa_split.json')
    assert [{'task_id':x['task_id'],'source_group':x['source_group']} for x in records] == split['support']
    assert len(records)==318 and len({x['task_id'] for x in records})==318
    assert not {x['task_id'] for x in records} & {x['task_id'] for x in split['future']}
    fields={'task_id','source_group','question','query','split','executor_id','executor_label',
        'history_executor_label','environment_id','context_sha256','selection_sha256','arms'}
    arms=[]
    for record in records:
        assert set(record)==fields
        assert record['executor_id']==settings['executor_id']==manifest['executor_id']
        assert record['executor_label']==settings['executor_label']
        assert record['split']=='train' and len(record['query'])==512
        assert torch.isfinite(torch.tensor(record['query'])).all()
        assert len(record['arms'])==2
        for arm in record['arms']:
            assert set(arm)=={'skill_id','document_sha256','physical_id','outcome'}
            assert arm['outcome'] is None or type(arm['outcome']) is int and arm['outcome'] in (0,1)
        arms.extend(record['arms'])
    assert sum(x['outcome'] is None for x in arms)==unknown
    assert len({x['physical_id'] for x in arms})==636
    assert settings['provider']==executor and settings['review_mode']=='exact_quote'
    tensors=load_file(str(bundle/'fitted.safetensors'))
    assert {k:tuple(v.shape) for k,v in tensors.items()}=={'bias':(785,),'mean':(512,),'delta':(785,512)}
    assert all(torch.isfinite(v).all() for v in tensors.values())
    assert sha(bundle/'fitted.safetensors')==manifest['source_bindings']['fitted.safetensors']['file_sha256']


@pytest.mark.parametrize('executor,selected', [('gpt', ('0.0', '0.3')), ('deepseek', ('0.05', '0.2'))])
def test_published_keyopt_fit_is_hash_bound_and_support_scoped(executor, selected):
    from safetensors import safe_open
    root = RELEASE_ROOT/'data/toolqa'; provenance = read(root/'provenance.json')
    entry = provenance['models'][executor]
    path = root/executor/'keyopt.safetensors'
    assert sha(path) == entry['files']['keyopt.safetensors']['sha256'] == entry['keyopt']['source_sha256']
    with safe_open(str(path), 'np') as f:
        shapes = {k: f.get_tensor(k).shape for k in f.keys()}; meta = f.metadata()
    assert shapes == {'beta': (785,), 'delta': (785, 512), 'mu': (512,)}
    assert (meta['alpha'], meta['score_cap'], meta['use_residual']) == (*selected, 'True')
    scores = root/'support_reranker_scores.json'
    assert sha(scores) == provenance['shared_files']['support_reranker_scores.json']['sha256']
    split = read(RELEASE_ROOT/'configs/toolqa_split.json'); scores = read(scores)
    assert set(scores) == {x['task_id'] for x in split['support']}
    assert not set(scores) & {x['task_id'] for x in split['future']}
    for record in read(root/executor/'support.json'):
        assert set(scores[record['task_id']]) == {a['skill_id'] for a in record['arms']}
