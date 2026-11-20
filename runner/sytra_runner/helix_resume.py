"""Atomic step-boundary checkpoints for the experimental Helix SFT loop."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch

from .helix_checkpoint import _checkpoint_sources, save_mapped_adapter


def _batch_digest(batches) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        batch.supervised_tokens()
        tensors = (batch.input_ids, batch.attention_mask, batch.labels)
        if hasattr(batch, 'cu_seqlens'):
            digest.update(b'helix-packed-boundaries-v1')
            tensors += (batch.cu_seqlens,)
        for tensor in tensors:
            value = tensor.detach().contiguous()
            digest.update(json.dumps([str(value.dtype), list(value.shape)]).encode('utf-8'))
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _base_identity(model) -> dict:
    base = Path(model.peft_config['default'].base_model_name_or_path).resolve(strict=True)
    paths = {path for path, _ in _checkpoint_sources(base).values()}
    paths.add(base / 'config.json')
    index = base / 'model.safetensors.index.json'
    if index.is_file():
        paths.add(index)
    return {'path': str(base), 'files': [
        [str(path.relative_to(base)), path.stat().st_size, path.stat().st_mtime_ns,
         hashlib.sha256(path.read_bytes()).hexdigest() if path.suffix == '.json' else None]
        for path in sorted(paths)
    ]}


def _optimizer_groups(model, optimizer) -> list[list[str]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    seen = set()
    for group in optimizer.param_groups:
        group_names = []
        for parameter in group['params']:
            if id(parameter) not in names or not parameter.requires_grad or id(parameter) in seen:
                raise ValueError('Optimizer must contain distinct trainable model parameters only')
            seen.add(id(parameter))
            group_names.append(names[id(parameter)])
        groups.append(group_names)
    if seen != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError('Optimizer must include every trainable parameter')
    return groups


def _signature(model, optimizer, scheduler, batches, accumulation_steps, device, identity):
    if not batches or type(accumulation_steps) is not int or accumulation_steps <= 0:
        raise ValueError('Checkpoint requires batches and positive accumulation steps')
    device = torch.device(device)
    if device.type not in {'cpu', 'cuda'} or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Checkpoint device is unavailable')
    return {'base': _base_identity(model), 'batch_digest': _batch_digest(batches),
            'accumulation_steps': accumulation_steps, 'device': str(device),
            'optimizer': type(optimizer).__module__ + '.' + type(optimizer).__qualname__,
            'optimizer_groups': _optimizer_groups(model, optimizer),
            'scheduler': None if scheduler is None else type(scheduler).__module__ + '.' + type(scheduler).__qualname__,
            'identity': json.loads(json.dumps(identity))}


def save_helix_checkpoint(path, model, optimizer, scheduler, batches, *, step, accumulation_steps,
                          device='cpu', identity=None):
    """Save an adapter and training state atomically at an optimizer boundary."""
    if type(step) is not int or step <= 0:
        raise ValueError('Checkpoint step must be positive')
    if any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad):
        raise ValueError('Checkpoint requires cleared gradients at an optimizer boundary')
    signature = _signature(model, optimizer, scheduler, batches, accumulation_steps, device, identity)
    target = Path(path).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f'Checkpoint already exists: {target}')
    staged = Path(tempfile.mkdtemp(prefix='.helix-checkpoint-', dir=target.parent))
    try:
        if hasattr(model, 'peft_config') and (getattr(model, 'sytra_nf4_format', None) is not None or getattr(model, 'sytra_quant_format', None) is not None):
            save_mapped_adapter(model, staged)
        else:
            model.save_pretrained(str(staged), safe_serialization=True, save_embedding_layers=False)
        torch.save(optimizer.state_dict(), staged / 'optimizer.pt')
        torch.save(None if scheduler is None else scheduler.state_dict(), staged / 'scheduler.pt')
        rng = {'cpu': torch.random.get_rng_state()}
        if torch.device(device).type == 'cuda':
            rng['cuda'] = torch.cuda.get_rng_state(device)
        torch.save(rng, staged / 'rng_state.pth')
        (staged / 'trainer_state.json').write_text(json.dumps({'global_step': step, 'helix': signature}, indent=2), encoding='utf-8')
        os.rename(staged, target)
    finally:
        if staged.exists():
            shutil.rmtree(staged)


def restore_helix_training_state(path, model, optimizer, scheduler, batches, *, accumulation_steps,
                                 device='cpu', identity=None) -> int:
    """Restore optimizer, scheduler and PyTorch RNG after loading the adapter."""
    root = Path(path)
    state = json.loads((root / 'trainer_state.json').read_text(encoding='utf-8'))
    step = state.get('global_step')
    if type(step) is not int or step <= 0:
        raise ValueError('Invalid Helix checkpoint step')
    expected = _signature(model, optimizer, scheduler, batches, accumulation_steps, device, identity)
    if state.get('helix') != expected:
        raise ValueError('Helix checkpoint configuration, dataset, optimizer or base model changed')
    optimizer_state = torch.load(root / 'optimizer.pt', map_location='cpu', weights_only=True)
    scheduler_state = torch.load(root / 'scheduler.pt', map_location='cpu', weights_only=True)
    rng = torch.load(root / 'rng_state.pth', map_location='cpu', weights_only=True)
    if not isinstance(rng, dict) or not isinstance(rng.get('cpu'), torch.Tensor):
        raise ValueError('Invalid Helix RNG state')
    if torch.device(device).type == 'cuda' and not isinstance(rng.get('cuda'), torch.Tensor):
        raise ValueError('Helix checkpoint has no CUDA RNG state')
    if (scheduler is None) != (scheduler_state is None):
        raise ValueError('Helix scheduler does not match checkpoint')
    optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
    torch.random.set_rng_state(rng['cpu'])
    if torch.device(device).type == 'cuda':
        torch.cuda.set_rng_state(rng['cuda'], device)
    return step
