"""Exercise the opt-in runner with a local tokenizer, model and dataset."""
import json

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
pytest.importorskip('peft')
pytest.importorskip('datasets')
tokenizers = pytest.importorskip('tokenizers')
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from safetensors.torch import load_file

from sytra_runner.backends.helix_sft import run_helix_training
from sytra_runner.config import RunConfig
from test_helix_mixtral import make_model
from test_performance import config_raw


def local_checkpoint(path, *, packed=False):
    if packed:
        from transformers import MixtralConfig, MixtralForCausalLM
        model = MixtralForCausalLM(MixtralConfig(vocab_size=32, hidden_size=64, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, num_local_experts=3,
            num_experts_per_tok=2, max_position_embeddings=64, use_cache=False, attention_dropout=0.0,
            sliding_window=None))
    else:
        model = make_model(top_k=2, experts=3)
        for layer in model.model.layers:
            layer.self_attn.q_proj = layer.self_attn.q_proj.base_layer
    model.save_pretrained(str(path))
    vocab = {'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, '<|user|>': 3, '<|assistant|>': 4,
             'hello': 5, 'world': 6, 'bye': 7, 'done': 8}
    tokenizer = tokenizers.Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]',
        eos_token='[EOS]', additional_special_tokens=['<|user|>', '<|assistant|>'])
    tokenizer.chat_template = "{% for m in messages %}{{ '<|' + m['role'] + '|>' }} {{ m['content'] }} {% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }} {% endif %}"
    tokenizer.save_pretrained(str(path))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_runner_packed_training_and_resume(tmp_path):
    base = tmp_path / 'base'
    local_checkpoint(base, packed=True)
    data = tmp_path / 'train.jsonl'
    data.write_text('\n'.join(json.dumps({'prompt': p, 'completion': c}) for p, c in
                              [('hello', 'world'), ('bye', 'done'), ('hello', 'done')]) + '\n')
    validation = tmp_path / 'validation.jsonl'
    validation.write_text(json.dumps({'prompt': 'bye', 'completion': 'world'}) + '\n')
    raw = config_raw(tmp_path)
    raw.update(model=str(base), train_dataset=str(data), val_dataset=str(validation), backend={'kind': 'cuda'},
               performance={'training_engine': 'helix', 'cpu_threads': 2})
    raw['adapter'].update(type='lora', rank=2, alpha=4, dropout=0, quant_bits=None)
    raw['train'].update(max_steps=2, epochs=None, batch_size=1, save_every=1, max_seq_len=32, packing=True)
    raw['optim'].update(grad_accumulation_steps=1, warmup_steps=0)
    assert run_helix_training(RunConfig.from_dict(raw)) == 0
    output = tmp_path / 'new-run'
    state = json.loads((output / 'checkpoint-1' / 'trainer_state.json').read_text())
    assert state['helix']['identity']['packing'] == 'flex_attention_v1'
    assert len(state['helix']['identity']['validation_digest']) == 64
    baseline = load_file(str(output / 'adapter_model.safetensors'))
    raw['output'] = {'adapter_path': str(tmp_path / 'resumed'),
                     'resume_from_checkpoint': str(output / 'checkpoint-1')}
    assert run_helix_training(RunConfig.from_dict(raw)) == 0
    resumed = load_file(str(tmp_path / 'resumed' / 'adapter_model.safetensors'))
    for name in baseline:
        torch.testing.assert_close(baseline[name], resumed[name], atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('backend', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('topology,quantized', [('attention', False), ('moe_hybrid', False),
                                                 ('moe_full', False), ('moe_full', True)])
def test_runner_training_and_checkpoint_resume(tmp_path, backend, topology, quantized, capsys):
    base = tmp_path / 'base'
    local_checkpoint(base)
    data = tmp_path / 'train.jsonl'
    data.write_text('\n'.join(json.dumps({'prompt': prompt, 'completion': answer}) for prompt, answer in
                              [('hello', 'world'), ('bye', 'done'), ('hello', 'done')]) + '\n')
    raw = config_raw(tmp_path)
    raw.update(model=str(base), train_dataset=str(data), backend={'kind': backend},
               performance={'training_engine': 'helix', 'cpu_threads': 2})
    if topology == 'attention' and backend == 'cpu':
        validation = tmp_path / 'validation.jsonl'
        validation.write_text(json.dumps({'prompt': 'bye', 'completion': 'world'}) + '\n')
        raw['val_dataset'] = str(validation)
    if topology == 'moe_full':
        raw['performance']['activation_memory_mb'] = 0
        if backend == 'cuda':
            raw['performance']['helix_prefetch'] = True
    raw['adapter'].update(type='qlora' if quantized else 'lora', rank=2, alpha=4, dropout=.1,
                          quant_bits=4 if quantized else None, moe_topology=topology)
    raw['train'].update(max_steps=2, epochs=None, batch_size=1, save_every=1, max_seq_len=32,
                        packing=False, neftune_noise_alpha=5.0)
    raw['optim'].update(grad_accumulation_steps=2, warmup_steps=0)
    if topology != 'attention':
        raw['optim']['router_lr_ratio'] = .1
        raw['optim']['router_aux_coef'] = .03
    assert run_helix_training(RunConfig.from_dict(raw)) == 0
    output = tmp_path / 'new-run'
    baseline = load_file(str(output / 'adapter_model.safetensors'))
    assert (output / 'sytra-helix-nf4.json').is_file() == quantized
    assert (output / 'checkpoint-1' / 'sytra-helix-nf4.json').is_file() == quantized
    if quantized:
        assert list((output / '.helix-nf4-cache').glob('*.safetensors'))
    if topology != 'attention':
        assert any('.gate.router_projection.lora_B.' in name for name in baseline)
        groups = torch.load(output / 'checkpoint-1' / 'optimizer.pt', weights_only=True)['param_groups']
        assert groups[1]['lr'] == pytest.approx(groups[0]['lr'] * .1)
    assert any('.experts.' in name for name in baseline) == (topology == 'moe_full')
    assert (output / 'checkpoint-1' / 'trainer_state.json').is_file()
    state = json.loads((output / 'checkpoint-1' / 'trainer_state.json').read_text())
    assert state['helix']['identity']['neftune_noise_alpha'] == 5.0
    if raw.get('val_dataset'):
        assert len(state['helix']['identity']['validation_digest']) == 64
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
        evals = [line for line in events if line.get('event') == 'eval']
        assert [line['payload']['step'] for line in evals] == [1, 2]
        assert all(line['payload']['tokens'] > 0 for line in evals)
    raw['output'] = {'adapter_path': str(tmp_path / 'resumed'),
                     'resume_from_checkpoint': str(output / 'checkpoint-1')}
    raw['optim'].pop('router_lr_ratio', None)
    raw['optim'].pop('router_aux_coef', None)
    assert run_helix_training(RunConfig.from_dict(raw)) == 0
    resumed = load_file(str(tmp_path / 'resumed' / 'adapter_model.safetensors'))
    assert set(baseline) == set(resumed)
    for name in baseline:
        torch.testing.assert_close(baseline[name], resumed[name], atol=1e-7, rtol=1e-6)
    assert not list(output.glob('.helix-activations-*'))
    assert not list((tmp_path / 'resumed').glob('.helix-activations-*'))
    if topology == 'attention' and backend == 'cpu':
        raw['train']['neftune_noise_alpha'] = 4.0
        raw['output']['adapter_path'] = str(tmp_path / 'wrong-noise')
        with pytest.raises(ValueError, match='configuration, dataset'):
            run_helix_training(RunConfig.from_dict(raw))
    if topology == 'moe_full':
        raw['output']['adapter_path'] = str(tmp_path / 'wrong-precision')
        raw['adapter'].update(type='lora' if quantized else 'qlora', quant_bits=None if quantized else 4)
        with pytest.raises(ValueError, match='same Helix base quantization'):
            run_helix_training(RunConfig.from_dict(raw))


def test_runner_rejects_unsupported_configuration(tmp_path):
    raw = config_raw(tmp_path)
    raw['performance'] = {'training_engine': 'helix'}
    raw['adapter']['quant_bits'] = 4
    with pytest.raises(ValueError, match='unquantized'):
        run_helix_training(RunConfig.from_dict(raw))


def test_dispatch_selects_helix_without_loading_unsloth(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import yaml
    from sytra_runner.backends import unsloth_sft
    raw = config_raw(tmp_path)
    raw['performance'] = {'training_engine': 'helix'}
    path = tmp_path / 'run.yaml'
    path.write_text(yaml.safe_dump(raw))
    calls = []
    monkeypatch.setitem(sys.modules, 'sytra_runner.backends.helix_sft',
                        SimpleNamespace(run_helix_training=lambda config: calls.append(config) or 0))
    assert unsloth_sft.train(str(path)) == 0
    assert len(calls) == 1 and calls[0].performance['training_engine'] == 'helix'
