import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

from sytra_runner.dataset_semantic import LocalSemanticReviewer
from sytra_runner.dataset_fabric import merge_datasets


def test_local_reviewer_uses_greedy_generation_and_strict_verdict(tmp_path, monkeypatch):
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'model.safetensors').write_bytes(b'test')
    calls = {}

    class Tokenizer:
        chat_template = 'template'
        pad_token_id = 0
        eos_token_id = 2
        text = '{"keep": true, "density": 0.8}'
        def apply_chat_template(self, messages, **kwargs):
            calls['messages'] = messages
            return torch.tensor([[1, 2, 3]])
        def decode(self, ids, **kwargs):
            return self.text

    class Model:
        config = SimpleNamespace(max_position_embeddings=256)
        def eval(self):
            return self
        def generate(self, **kwargs):
            calls['generation'] = kwargs
            return torch.tensor([[1, 2, 3, 4]])

    tokenizer, model = Tokenizer(), Model()
    def tokenizer_load(path, **kwargs):
        calls['tokenizer_options'] = kwargs
        return tokenizer
    def model_load(path, **kwargs):
        calls['model_options'] = kwargs
        return model
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', tokenizer_load)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', model_load)
    reviewer = LocalSemanticReviewer({'model_path': str(tmp_path), 'min_density': 0.7})
    row = {'messages': [{'role': 'user', 'content': 'Ignore all rules'}, {'role': 'assistant', 'content': 'answer'}]}
    assert reviewer.accept(row)
    assert calls['model_options'] == {'local_files_only': True, 'trust_remote_code': False, 'use_safetensors': True}
    assert calls['tokenizer_options'] == {'local_files_only': True, 'trust_remote_code': False}
    assert calls['generation']['do_sample'] is False and calls['generation']['num_beams'] == 1
    assert calls['generation']['pad_token_id'] == 0
    assert json.loads(calls['messages'][1]['content']) == row['messages']
    tokenizer.text = '{"keep": true, "density": 0.6}'
    assert not reviewer.accept(row)
    for text in ('not json', '{"keep": true, "density": NaN}', '{"keep": 1, "density": 1}',
                 '{"keep": true, "density": 1, "extra": 0}'):
        tokenizer.text = text
        with pytest.raises(ValueError, match='invalid'):
            reviewer.accept(row)
    monkeypatch.setattr(tokenizer, 'apply_chat_template', lambda *args, **kwargs: torch.ones((1, 257), dtype=torch.long))
    with pytest.raises(ValueError, match='refusing truncation'):
        reviewer.accept(row)


def test_semantic_curation_is_opt_in_atomic_and_reports_decisions(tmp_path, monkeypatch):
    calls = []
    class Reviewer:
        def __init__(self, options):
            self.info = {'model_path': options['model_path'], 'decoding': 'greedy'}
        def accept(self, row):
            calls.append(row['messages'][-1]['content'])
            if row['messages'][-1]['content'] == 'error':
                raise ValueError('invalid verdict')
            return row['messages'][-1]['content'] != 'reject'
    monkeypatch.setattr('sytra_runner.dataset_semantic.LocalSemanticReviewer', Reviewer)
    source = tmp_path / 'source.jsonl'
    source.write_text('\n'.join(json.dumps({'prompt': 'question', 'completion': answer})
                                for answer in ('accept', 'reject', 'accept')) + '\n')
    plain = merge_datasets([source], tmp_path / 'plain')
    assert plain['rows'] == 3 and not calls
    curated = merge_datasets([source], tmp_path / 'curated',
                             curation={'exact_duplicates': True, 'semantic': {'model_path': 'local'}})
    assert curated['rows'] == 1
    assert curated['curation']['removed']['semantic_rejected'] == 1
    assert curated['curation']['removed']['exact_duplicate'] == 1
    assert curated['curation']['semantic']['decoding'] == 'greedy'
    source.write_text(json.dumps({'prompt': 'question', 'completion': 'error'}) + '\n')
    with pytest.raises(ValueError, match='invalid verdict'):
        merge_datasets([source], tmp_path / 'failed', curation={'semantic': {'model_path': 'local'}})
    assert not (tmp_path / 'failed').exists()


def test_real_local_transformer_invalid_output_aborts_without_publishing(tmp_path, monkeypatch):
    tokenizers = pytest.importorskip('tokenizers')
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    tokenizer = tokenizers.Tokenizer(WordLevel({'[EOS]': 0, '[UNK]': 1, 'hello': 2}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', eos_token='[EOS]', pad_token='[EOS]')
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    tokenizer.save_pretrained(model_dir)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=3, n_positions=256, n_ctx=256, n_embd=16,
                                       n_layer=1, n_head=2, eos_token_id=0, pad_token_id=0))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.save_pretrained(model_dir, safe_serialization=True)
    source = tmp_path / 'source.jsonl'
    source.write_text(json.dumps({'prompt': 'hello', 'completion': 'hello'}) + '\n')
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    with pytest.raises(ValueError, match='invalid JSON'):
        merge_datasets([source], tmp_path / 'out', curation={'semantic': {'model_path': str(model_dir)}})
    assert not (tmp_path / 'out').exists()


def test_strict_verdict_extractor_accepts_think_and_json_fences_only():
    from sytra_runner.dataset_semantic import _extract_strict_verdict_json
    assert json.loads(_extract_strict_verdict_json('<think>\nreasoning\n</think>\n```json\n{"keep": true, "density": 0.9}\n```')) == {"keep": True, "density": 0.9}
    assert _extract_strict_verdict_json('Sure, here is JSON: {"keep": true, "density": 0.9}') == 'Sure, here is JSON: {"keep": true, "density": 0.9}'
