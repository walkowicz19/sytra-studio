"""Compare real LoRA updates with mapped Helix and CUDA-resident Transformers."""

import argparse
import gc
from itertools import groupby
import json
import math
from pathlib import Path
from statistics import median
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
import transformers
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import MixtralConfig, MixtralForCausalLM, PreTrainedTokenizerFast

from sytra_runner.backends.unsloth_sft import _assistant_masked_row
from sytra_runner.helix_checkpoint import _checkpoint_sources, load_mapped_moe
from sytra_runner.helix_sft import HelixDatasetBatches, train_helix_batches

SEED = 3407
TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
BATCH_SIZE = 4
SIZES = {
    'tiny': (4, 512, 1024, 8),
    '1b': (16, 1536, 3072, 12),
}


def save_scaled_checkpoint(checkpoint: Path, config: MixtralConfig) -> int:
    """Materialize one shard at a time to bound host RAM during model creation."""
    with torch.device('meta'):
        model = MixtralForCausalLM(config)
    config.save_pretrained(checkpoint)
    state = model.state_dict()
    weight_map = {}
    for index, (_, names) in enumerate(groupby(
        state, key=lambda name: name.split('.')[2] if name.startswith('model.layers.') else name,
    )):
        shard = {}
        for name in names:
            tensor = torch.empty_like(state[name], device='cpu')
            shard[name] = tensor.fill_(1) if name.endswith('norm.weight') else tensor.normal_(0, config.initializer_range)
        filename = f'model-{index + 1:05}.safetensors'
        save_file(shard, checkpoint / filename, metadata={'format': 'pt'})
        weight_map.update(dict.fromkeys(shard, filename))
    size_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state.values())
    (checkpoint / 'model.safetensors.index.json').write_text(
        json.dumps({'metadata': {'total_size': size_bytes}, 'weight_map': weight_map}), encoding='utf-8',
    )
    return sum(parameter.numel() for parameter in model.parameters())


def save_helix_layout(source: Path, destination: Path) -> None:
    """Store separate expert tensors to avoid native fused slice reads on Windows."""
    source_config = MixtralConfig.from_pretrained(source, local_files_only=True)
    source_config.save_pretrained(destination)
    weight_map = {}
    total_size = 0

    def write_tensor(name, tensor):
        nonlocal total_size
        filename = f'model-{len(weight_map) + 1:05}.safetensors'
        save_file({name: tensor.contiguous()}, destination / filename, metadata={'format': 'pt'})
        weight_map[name] = filename
        total_size += tensor.numel() * tensor.element_size()

    for file_path, entries in groupby(_checkpoint_sources(source).values(), key=lambda item: item[0]):
        with safe_open(file_path, framework='pt', device='cpu') as shard:
            for _, name in entries:
                tensor = shard.get_tensor(name)
                if name.endswith('.experts.gate_up_proj'):
                    prefix = name.removesuffix('.gate_up_proj')
                    for expert, weights in enumerate(tensor):
                        gate, up = weights.chunk(2, dim=0)
                        write_tensor(f'{prefix}.{expert}.w1.weight', gate)
                        write_tensor(f'{prefix}.{expert}.w3.weight', up)
                elif name.endswith('.experts.down_proj'):
                    prefix = name.removesuffix('.down_proj')
                    for expert, weights in enumerate(tensor):
                        write_tensor(f'{prefix}.{expert}.w2.weight', weights)
                else:
                    write_tensor(name, tensor)

    (destination / 'model.safetensors.index.json').write_text(
        json.dumps({'metadata': {'total_size': total_size}, 'weight_map': weight_map}), encoding='utf-8',
    )


def prepare_checkpoint(checkpoint: Path, size: str) -> int:
    torch.manual_seed(SEED)
    layers, hidden, intermediate, heads = SIZES[size]
    config = MixtralConfig(
        vocab_size=512, hidden_size=hidden, intermediate_size=intermediate, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=heads // 2, num_local_experts=4,
        num_experts_per_tok=2, max_position_embeddings=128, use_cache=False,
        sliding_window=None, tie_word_embeddings=False,
    )
    if size == 'tiny':
        model = MixtralForCausalLM(config)
        model.save_pretrained(checkpoint)
        return sum(p.numel() for p in model.parameters())
    return save_scaled_checkpoint(checkpoint, config)


def prepare_batches(dataset_path: Path):
    examples = [
        {'prompt': f'Double the number {n} and explain the result in a complete sentence.',
         'completion': f'The answer is {2 * n} because adding {n} to itself gives the number {2 * n}.'}
        for n in range(64)
    ]
    dataset_path.write_text(''.join(json.dumps(row) + '\n' for row in examples), encoding='utf-8')
    words = sorted({word for row in examples for value in row.values() for word in value.split()})
    specials = ['[UNK]', '[PAD]', '[EOS]', '<|user|>', '<|assistant|>']
    vocabulary = {word: index for index, word in enumerate(specials + ['.', ','] + words)}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]',
        eos_token='[EOS]', additional_special_tokens=specials[3:],
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ '<|' + m['role'] + '|>' }} {{ m['content'] }} "
        "{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }} {% endif %}"
    )
    rows = [_assistant_masked_row(json.loads(line), tokenizer, 64)
            for line in dataset_path.read_text(encoding='utf-8').splitlines()]
    return list(HelixDatasetBatches(rows, BATCH_SIZE, tokenizer.pad_token_id))


def measure(engine, checkpoint, batches, warmup, steps):
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    if engine == 'helix':
        model, pager = load_mapped_moe(checkpoint, rank=8, alpha=16, target_modules=TARGETS)
    else:
        base = MixtralForCausalLM.from_pretrained(checkpoint, attn_implementation='sdpa')
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=list(TARGETS))).cuda()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0)
    durations, losses = [], []
    measured_tokens = 0
    torch.cuda.synchronize()
    started = perf_counter()

    def record(step, loss, tokens):
        nonlocal started, measured_tokens
        torch.cuda.synchronize()
        now = perf_counter()
        if step > warmup:
            durations.append(now - started)
            losses.append(loss)
            measured_tokens += tokens
        started = now

    if engine == 'helix':
        train_helix_batches(model, pager, batches, optimizer, max_steps=warmup + steps,
                            device='cuda', on_step=record)
    else:
        for step in range(1, warmup + steps + 1):
            batch = batches[(step - 1) % len(batches)]
            loss = model(input_ids=batch.input_ids.cuda(), attention_mask=batch.attention_mask.cuda(),
                         labels=batch.labels.cuda(), use_cache=False).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            record(step, loss.item(), batch.supervised_tokens())

    if len(durations) != steps or not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError(f'{engine} produced incomplete or non-finite measurements')
    return {
        'engine': engine, 'step_seconds': durations, 'median_step_seconds': median(durations),
        'supervised_tokens': measured_tokens, 'supervised_tokens_per_second': measured_tokens / sum(durations),
        'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
        'trainable_parameters': sum(p.numel() for p in trainable),
        'first_measured_loss': losses[0], 'last_measured_loss': losses[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', choices=SIZES, default='tiny')
    parser.add_argument('--vram-budget-mib', type=int, help='Optional per-process PyTorch CUDA allocator limit')
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--output', type=Path, help='Optional path to save the JSON result')
    args = parser.parse_args()
    if args.warmup < 1 or args.steps < 1:
        parser.error('warmup and measured steps must be positive')
    if not torch.cuda.is_available():
        parser.error('CUDA is required')

    torch.set_num_threads(4)
    gpu_bytes = torch.cuda.get_device_properties(0).total_memory
    if args.vram_budget_mib is not None:
        budget_bytes = args.vram_budget_mib * 2**20
        if not 0 < budget_bytes <= gpu_bytes:
            parser.error('VRAM budget must be positive and no larger than total GPU memory')
        torch.cuda.set_per_process_memory_fraction(budget_bytes / gpu_bytes, device=0)
    with TemporaryDirectory(prefix='helix-training-', dir=Path.cwd()) as tmp:
        checkpoint = Path(tmp) / 'model'
        print(f'Preparing Mixtral {args.size} checkpoint', file=sys.stderr, flush=True)
        parameters = prepare_checkpoint(checkpoint, args.size)
        batches = prepare_batches(Path(tmp) / 'train.jsonl')
        helix_checkpoint = checkpoint
        if args.size == '1b':
            print('Preparing separate expert tensors for Helix', file=sys.stderr, flush=True)
            helix_checkpoint = Path(tmp) / 'helix-model'
            save_helix_layout(checkpoint, helix_checkpoint)
        weight_bytes = parameters * torch.empty((), dtype=torch.float32).element_size()
        order = ('helix', 'traditional', 'traditional', 'helix')
        if args.vram_budget_mib is not None:
            order = ('helix', 'traditional')
        traditional_skip = None
        traditional_oom = None
        if weight_bytes > gpu_bytes:
            traditional_skip = 'FP32 weights alone exceed total physical GPU memory.'
            print(traditional_skip, file=sys.stderr, flush=True)
            order = tuple(engine for engine in order if engine == 'helix')
        trials = []
        for number, engine in enumerate(order, start=1):
            print(f'Trial {number}/{len(order)}: {engine}', file=sys.stderr, flush=True)
            model_path = helix_checkpoint if engine == 'helix' else checkpoint
            try:
                trials.append(measure(engine, model_path, batches, args.warmup, args.steps))
            except torch.cuda.OutOfMemoryError as error:
                if engine != 'traditional' or args.vram_budget_mib is None:
                    raise
                traditional_oom = str(error)
                print('Traditional CUDA run exceeded the configured VRAM budget.', file=sys.stderr, flush=True)

    if len({(row['trainable_parameters'], row['supervised_tokens']) for row in trials}) != 1:
        raise RuntimeError('The workloads differ in trainable parameters or supervised tokens')
    report = {
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'model_parameters': parameters,
        'size': args.size, 'gpu_total_mib': gpu_bytes / 2**20, 'base_weight_mib': weight_bytes / 2**20,
        'traditional_skip': traditional_skip,
        'traditional_oom': traditional_oom, 'vram_budget_mib': args.vram_budget_mib,
        'precision': 'FP32', 'rows': 64, 'batch_size': BATCH_SIZE,
        'warmup_steps': args.warmup, 'measured_steps_per_trial': args.steps, 'trials': trials,
    }
    result = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding='utf-8')
    print(result, end='')


if __name__ == '__main__':
    main()
