import json

import pytest

pytest.importorskip('transformers')
tokenizers = pytest.importorskip('tokenizers')
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from sytra_runner.dataset_context import NativeContextFilter
from sytra_runner.dataset_fabric import merge_datasets


def make_tokenizer(path):
    path.mkdir()
    (path / 'config.json').write_text('{}')
    tokenizer = tokenizers.Tokenizer(WordLevel({'[UNK]': 0, 'user': 1, 'assistant': 2, 'hello': 3,
                                                'answer': 4, 'more': 5}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]')
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}"
    tokenizer.save_pretrained(path)


def test_context_filter_counts_native_chat_tokens_without_truncating(tmp_path, monkeypatch):
    model = tmp_path / 'tokenizer'
    make_tokenizer(model)
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    source = tmp_path / 'source.jsonl'
    rows = [{'prompt': 'hello', 'completion': 'answer'},
            {'prompt': 'hello more more', 'completion': 'answer more'}]
    source.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    plain = merge_datasets([source], tmp_path / 'plain')
    assert plain['rows'] == 2 and 'curation' not in plain
    result = merge_datasets([source], tmp_path / 'curated',
                            curation={'context': {'tokenizer_path': str(model), 'max_seq_length': 5}})
    assert result['rows'] == 1
    assert result['curation']['removed']['too_long'] == 1
    assert len(result['curation']['context']['files']['tokenizer.json']) == 64
    retained = json.loads((tmp_path / 'curated' / 'dataset.jsonl').read_text())
    assert retained['messages'] == [{'role': 'user', 'content': 'hello'},
                                    {'role': 'assistant', 'content': 'answer'}]


def test_context_filter_rejects_invalid_configuration(tmp_path):
    make_tokenizer(tmp_path / 'model')
    for options in ({'tokenizer_path': str(tmp_path / 'model'), 'max_seq_length': 1},
                    {'tokenizer_path': str(tmp_path / 'model'), 'max_seq_length': True},
                    {'tokenizer_path': str(tmp_path / 'model'), 'max_seq_length': 5, 'unknown': 1}):
        with pytest.raises(ValueError):
            NativeContextFilter(options)
