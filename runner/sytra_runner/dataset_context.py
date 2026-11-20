"""Opt-in native-tokenizer context limit for curated conversations."""
from __future__ import annotations

import hashlib
from pathlib import Path


class NativeContextFilter:
    def __init__(self, options):
        allowed = {'tokenizer_path', 'max_seq_length', 'min_tokens'}
        if not isinstance(options, dict) or set(options) - allowed or 'tokenizer_path' not in options:
            raise ValueError('Context filter requires tokenizer_path and max_seq_length')
        if 'max_seq_length' not in options and 'min_tokens' not in options:
            raise ValueError('Context filter requires tokenizer_path and max_seq_length')
        raw = options['tokenizer_path']
        limit = options.get('max_seq_length')
        min_tokens = options.get('min_tokens')
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError('Invalid native tokenizer path or context limit')
        if limit is not None and (type(limit) is not int or not 2 <= limit <= 1048576):
            raise ValueError('Invalid native tokenizer path or context limit')
        if min_tokens is not None and (type(min_tokens) is not int or not 0 <= min_tokens <= 1048576):
            raise ValueError('Invalid native tokenizer minimum token threshold')
        self.path = Path(raw).resolve(strict=True)
        if not self.path.is_dir() or not (self.path / 'config.json').is_file():
            raise ValueError('Context filter requires a local model tokenizer directory')
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True, trust_remote_code=False)
        if limit is not None and not self.tokenizer.chat_template:
            raise ValueError('Context filter requires a tokenizer chat template')
        self.limit = limit
        self.min_tokens = min_tokens or 0
        self.info = {'tokenizer_path': str(self.path), 'files': {}}
        if limit is not None:
            self.info['max_seq_length'] = limit
        if self.min_tokens:
            self.info['min_tokens'] = self.min_tokens
        for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                     'special_tokens_map.json', 'vocab.json', 'merges.txt', 'tokenizer.model'):
            path = self.path / name
            if path.is_file():
                digest = hashlib.sha256()
                with path.open('rb') as file:
                    for chunk in iter(lambda: file.read(1024 * 1024), b''):
                        digest.update(chunk)
                self.info['files'][name] = digest.hexdigest()

    def count_message_tokens(self, row) -> int:
        total = 0
        for message in row['messages']:
            ids = self.tokenizer.encode(message['content'], add_special_tokens=False)
            if hasattr(ids, 'tolist'):
                ids = ids.tolist()
            if not isinstance(ids, list) or any(type(token) is not int for token in ids):
                raise ValueError('Tokenizer must return a flat list of token IDs')
            total += len(ids)
        return total

    def count_template_tokens(self, row) -> int:
        ids = self.tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=False,
                                                 return_dict=False, truncation=False)
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        if not isinstance(ids, list) or any(type(token) is not int for token in ids):
            raise ValueError('Tokenizer must return a flat list of token IDs')
        return len(ids)

    def accept(self, row):
        if self.limit is None:
            return True
        return self.count_template_tokens(row) <= self.limit
