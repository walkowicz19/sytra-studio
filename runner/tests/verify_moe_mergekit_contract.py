"""Check Sytra's generated MoE configs against the installed MergeKit.

Run with the merge environment's Python; uses fake model metadata and never
downloads or loads model weights.
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from mergekit.common import ModelReference
from mergekit.moe import ALL_OUTPUT_ARCHITECTURES
from mergekit.moe.config import MoEMergeConfig

from sytra_runner.config import MergeConfig
from sytra_runner.merge import write_moe_config


def main():
    names = {item.name().lower(): item for item in ALL_OUTPUT_ARCHITECTURES}
    if 'qwen moe' not in names:
        raise RuntimeError('Installed MergeKit does not support Qwen MoE')
    raw = {'version': 1, 'merge_method': 'moe', 'base_model': 'org/base', 'dtype': 'float16',
           'models': [{'model': 'org/a', 'positive_prompts': ['code']},
                      {'model': 'org/b', 'positive_prompts': ['math']}],
           'moe': {'architecture': 'qwen', 'experts_per_token': 2,
                   'shared_experts': [{'source_model': 'org/shared', 'positive_prompts': ['general']} ]},
           'compat': {'verdict': 'green'}, 'output': {'model_path': 'unused'}}
    with tempfile.TemporaryDirectory(prefix='sytra-moe-contract-', dir='.') as directory:
        path = write_moe_config(MergeConfig.from_dict(raw), str(Path(directory) / 'config'))
        config = MoEMergeConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding='utf-8')))
        with patch.object(ModelReference, 'config', return_value=SimpleNamespace(model_type='qwen2')):
            assert names['qwen moe'].supports_config(config)
            assert not names['mixtral'].supports_config(config)
        assert config.shared_experts[0].positive_prompts == ['general']
    print('Installed MergeKit accepted the Qwen shared-expert routing config.')


if __name__ == '__main__':
    main()
