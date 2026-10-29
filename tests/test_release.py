import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from skill_experience.config import RELEASE_ROOT
from skill_experience.io import write, read, sha, seal, verify, text_sha
from skill_experience.retrieval.compressed_skill_key import CompressedSkillKey
from skill_experience.retrieval.source_span_citations import request, validate
from skill_experience.toolqa.original import REVIEW_SYSTEM
from skill_experience.toolqa.full import validate_public, build_memory
from skill_experience.toolqa.execute import complete_gate
from skill_experience.training.recipe import batch_masks, objective, lr_factor


def test_released_checkpoint_loads_and_is_unit_normalized():
    file = RELEASE_ROOT / 'models/compressor.safetensors'
    assert sha(file) == read(RELEASE_ROOT / 'models/config.json')['sha256']
    model = CompressedSkillKey().eval()
    model.load_state_dict(load_file(str(file)), strict=True)
    torch.manual_seed(6)
    with torch.no_grad():
        result = model(torch.randn(3, 4096))
    assert result.shape == (3, 512)
    assert torch.isfinite(result).all()
    assert torch.allclose(result.norm(dim=1), torch.ones(3), atol=1e-5)


def test_train_kernel_masks_duplicate_positives_and_has_gradients():
    rows = [dict(skill_ids=['a'], duplicate_skills=['a', 'b']), dict(skill_ids=['c'])]
    positive, valid = batch_masks(rows, ['a', 'b', 'c'])
    assert valid.tolist() == [[True, False, True], [True, True, True]]
    model = CompressedSkillKey(condition_width=8, width=4, layers=1, expansion=8)
    cond = torch.randn(3, 8)
    native = F.normalize(cond[:, :4], dim=-1)
    query = F.normalize(torch.randn(2, 4), dim=-1)
    loss, _, anchor = objective(model(cond), native, query, positive, valid)
    assert abs(float(anchor.detach())) < 1e-6
    loss.backward()
    assert model.output_projection.weight.grad.abs().sum() > 0
    assert lr_factor(0, 100) == .2
    assert lr_factor(4, 100) == 1.
    assert lr_factor(99, 100) == 0.
    # Five-epoch refit must not stretch a five-epoch schedule to zero.
    assert lr_factor(24, 100) > .8


def test_checkpoint_rejects_trainable_or_zero_inputs():
    model = CompressedSkillKey(condition_width=8, width=4, layers=1, expansion=8)
    with pytest.raises(ValueError): model(torch.randn(2, 8, requires_grad=True))
    with pytest.raises(ValueError): model(torch.zeros(2, 8))


def test_citation_attribution_and_invalid_fallback():
    docs = [dict(skill_id=s, document=d, document_sha256=text_sha(d)) for s, d in [('a', 'A\n'), ('b', 'B\n')]]
    prompt = [dict(role='system', content=REVIEW_SYSTEM), dict(role='user', content=json.dumps(dict(
        question='q', proposed_skill_id='a', baseline_skill_id='a', candidates=docs, history=[])))]
    req, catalog = request({'prompt': prompt})
    value = dict(skill_id='b', quote='[doc2:L00001]', evidence_ids=[], reason='applicable')
    assert validate(json.dumps(value), req['prompt'], catalog)['valid']
    value['quote'] = '[doc1:L00001]'
    invalid = validate(json.dumps(value), req['prompt'], catalog)
    assert invalid['skill_id'] == 'a' and invalid['fallback_applied']
    assert invalid['learning_eligible'] is False


def test_unknown_calibration_is_rejected():
    from skill_experience.toolqa.full import calibration
    assert calibration({}) == 'keyopt'
    assert calibration({'calibration': 'fixed_strength'}) == 'fixed_strength'
    with pytest.raises(ValueError):
        calibration({'calibration': 'unbounded'})


def test_target_labels_rejected_before_routing():
    tasks = [dict(task_id='q', source_group='g', question='Question', answer='secret')]
    skill = dict(skill_id='a', name='a', description='', body='A', document='A', document_sha256=text_sha('A'))
    with pytest.raises(ValueError, match='Only public'):
        validate_public(tasks, [skill])


def test_support_identity_unknown_and_group_exclusion():
    skills = [dict(skill_id=s, document_sha256=text_sha(s)) for s in ['a', 'b']]
    event = dict(task_id='support', source_group='shared', question='q', split='train', query=[1., 0.],
        executor_id='model:low', executor_label='Model low', environment_id='toolqa-official-react-785',
        selection_sha256='0'*64,
        arms=[dict(skill_id=s, document_sha256=text_sha(s), physical_id=s, outcome=y) for s, y in [('a', 1), ('b', 0)]])
    memory, events = build_memory(torch.eye(2), skills, [event], 'model:low')
    assert len(events) == 1
    _, details = memory.score(torch.tensor([1., 0.]), excluded_source_groups=['shared'], return_details=True)
    assert details['evidence_ids'] == []
    event['arms'][1]['outcome'] = None
    _, events = build_memory(torch.eye(2), skills, [event], 'model:low')
    assert events == []
    event['executor_id'] = 'other'
    with pytest.raises(ValueError, match='same executor'):
        build_memory(torch.eye(2), skills, [event], 'model:low')


def test_seals_reject_changed_files_and_path_escape(tmp_path):
    write(tmp_path/'a.json', {'frozen': True}); seal(tmp_path, 'seal.json', [tmp_path/'a.json'])
    verify(tmp_path, 'seal.json')
    (tmp_path/'a.json').write_text('{}')
    with pytest.raises(ValueError): verify(tmp_path, 'seal.json')
    write(tmp_path/'escape.json', {'files': {'../outside': '0'*64}})
    with pytest.raises(ValueError): verify(tmp_path, 'escape.json')
    with pytest.raises(FileExistsError): write(tmp_path/'a.json', {})


def test_scoring_gate_requires_every_task_and_cleanup(tmp_path):
    write(tmp_path/'tasks.json', [{'task_id': 'a'}, {'task_id': 'b'}])
    seal(tmp_path, 'prepare.seal.json', [tmp_path/'tasks.json'])
    write(tmp_path/'frozen_guidance.json', [])
    seal(tmp_path, 'review.seal.json', [tmp_path/'frozen_guidance.json'])
    with pytest.raises(FileNotFoundError): complete_gate(tmp_path)
    write(tmp_path/'receipts.json', [{'task_id': 'a', 'native_released': True}])
    seal(tmp_path, 'generation.seal.json', [tmp_path/'receipts.json'])
    with pytest.raises(ValueError, match='Incomplete'): complete_gate(tmp_path)


@pytest.mark.parametrize('calibration', ['keyopt', 'fixed_strength'])
@pytest.mark.parametrize('mode', ['exact_quote', 'source_line'])
def test_full_pipeline_freezes_a_complete_cited_document_offline(tmp_path, monkeypatch, mode, calibration):
    from types import SimpleNamespace
    from safetensors.torch import save_file
    import skill_experience.toolqa.full as full
    import skill_experience.toolqa.transport as transport
    source, run = tmp_path/'inputs', tmp_path/'run'
    source.mkdir()
    skills = [dict(skill_id=f's{i}', name=f'Skill {i}', description='', body=f'Document {i}\n',
                   document=f'Document {i}\n', document_sha256=text_sha(f'Document {i}\n')) for i in range(10)]
    tasks = [dict(task_id='q', source_group='q-group', question='Public question')]
    write(source/'tasks.json', tasks); write(source/'skills.json', skills); write(source/'support.json', [])
    write(source/'settings.json', dict(provider='gpt', model='model', executor_id='model:low',
                                      executor_label='GPT-5.5 (low reasoning)', review_max_tokens=4000, review_mode=mode,
                                      calibration=calibration))
    # Only this synthetic unit fixture replaces scope metadata; the real CLI stays fixed.
    fixture = tmp_path/'fixture'
    write(fixture/'configs/toolqa_bank.json', [{k:s[k] for k in ('skill_id','document_sha256')} for s in skills])
    write(fixture/'configs/toolqa_split.json', {'support': [], 'future': [{'task_id':'q','source_group':'q-group'}]})
    monkeypatch.setattr(full, 'RELEASE_ROOT', fixture)
    keys = F.normalize(torch.randn(10, 512), dim=-1)
    save_file({'keys':keys, 'queries':keys[:1].clone()}, str(source/'features.safetensors'))
    if calibration == 'keyopt':
        # A support-fitted offset for s9 outweighs its lower reranker score.
        beta = torch.zeros(10); beta[9] = 1.
        save_file({'beta':beta, 'delta':torch.zeros(10,512), 'mu':torch.zeros(512)}, str(source/'keyopt.safetensors'),
                  metadata={'alpha':'0.0', 'score_cap':'0.3', 'use_residual':'True'})
    else:
        save_file({'bias':torch.zeros(10), 'mean':torch.zeros(512), 'delta':torch.zeros(10,512)}, str(source/'fitted.safetensors'))
    full.prepare(SimpleNamespace(inputs=str(source), output=str(run)))
    write(run/'reranking/scores.json', {'q':{s['skill_id']:1-i*.01 for i,s in enumerate(skills)}})
    seal(run, 'rerank.seal.json', [run/'reranking/scores.json'])
    full.select(SimpleNamespace(run=str(run)))
    request = read(run/'review_requests.json')[0]
    payload = json.loads(request['prompt'][1]['content'])
    assert len(payload['candidates']) == 8 and payload['history'] == []
    assert payload['proposed_skill_id'] == ('s9' if calibration == 'keyopt' else 's0')
    # Source-line IDs follow candidate order; cite s0 wherever the calibration placed it.
    s0_line = '[doc%d:L00001]' % ([c['skill_id'] for c in payload['candidates']].index('s0') + 1)
    class FakeClient:
        usage = [{'completion_tokens': 1}]
        def __init__(self, *a, **k): pass
        def reset_budget(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def __call__(self, *a):
            return json.dumps(dict(skill_id='s0', quote='Document 0' if mode=='exact_quote' else s0_line, evidence_ids=[], reason='matched'))
    monkeypatch.setattr(transport, 'Client', FakeClient)
    full.review(SimpleNamespace(run=str(run)))
    verify(run, 'review.seal.json')
    assert read(run/'frozen_guidance.json') == [dict(task_id='q', skill_id='s0',
        guidance='# Selected public skill documents\n\n## s0\nDocument 0\n')]


@pytest.mark.parametrize('module', ['training.compressor', 'toolqa.full', 'toolqa.execute'])
def test_cli_help_has_no_runtime_model_or_data_dependency(module, tmp_path):
    result = subprocess.run([sys.executable, '-m', 'skill_experience.'+module, '--help'],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout
    assert list(tmp_path.iterdir()) == []
