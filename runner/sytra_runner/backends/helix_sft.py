"""SFT runner for locally mapped/quantized multi-architecture MoE checkpoints (Helix)."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from .. import telemetry


def _process_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _causal_masked_row(row: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    """Tokenize full conversation with native chat template for full-sequence causal loss."""
    from .unsloth_sft import _to_prompt_completion
    converted = _to_prompt_completion(row)
    messages = converted['prompt'] + converted['completion']
    if any(message['role'] not in {'system', 'user', 'assistant'} for message in messages):
        raise ValueError('Unsupported conversation role for SFT')
    if any(message['role'] == 'assistant' and not message['content'].strip() for message in messages):
        raise ValueError('Assistant responses must not be empty')
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False)
    if hasattr(full, 'tolist'):
        full = full.tolist()
    if not isinstance(full, list) or any(type(token) is not int for token in full) or len(full) < 2:
        raise ValueError('Chat template must return at least two token IDs')
    if len(full) > max_length:
        raise ValueError(f'Conversation has {len(full)} tokens and exceeds max_seq_len={max_length}; refusing truncation')
    mask = [0] + [1] * (len(full) - 1)
    return {'input_ids': full, 'assistant_masks': mask}


def _validate(config):
    if config.train_mode != 'sft':
        raise ValueError('Helix supports SFT training only')
    if config.backend_kind not in {'auto', 'cuda', 'cpu'}:
        raise ValueError('Helix supports CPU and CUDA only')
    adapter_type = config.adapter.get('type')
    bits = config.adapter.get('quant_bits')
    if not ((adapter_type == 'lora' and bits in (None, 16)) or (adapter_type == 'qlora' and bits in (4, 8))):
        raise ValueError('Helix requires unquantized LoRA or 4-bit/8-bit QLoRA')
    if (config.train.get('packing') is True or config.performance.get('packing') is True) and config.train.get('batch_size', 2) != 1:
        raise ValueError('Helix packed training requires batch_size=1')
    epochs = config.train.get('epochs')
    max_steps = config.train.get('max_steps')
    if (epochs is None and not max_steps) or (epochs is not None and epochs <= 0) or (max_steps is not None and max_steps <= 0):
        raise ValueError('Helix requires a positive max_steps or positive epochs count')


_PAGED_ADAMW8BIT_CLS = None


def _get_helix_paged_adamw8bit_cls():
    """Return a PagedAdamW8bit subclass that streams CPU-resident Helix parameters through CUDA."""
    global _PAGED_ADAMW8BIT_CLS
    if _PAGED_ADAMW8BIT_CLS is not None:
        return _PAGED_ADAMW8BIT_CLS
    import torch
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_F
    from bitsandbytes.utils import sync_gpu

    class HelixPagedAdamW8bit(bnb.optim.PagedAdamW8bit):
        """PagedAdamW8bit adapter that stages CPU-resident LoRA weights one tensor at a time to CUDA."""

        def __init__(self, params, *, cuda_device: str | torch.device = 'cuda', **kwargs):
            dev = torch.device(cuda_device)
            if dev.type != 'cuda':
                raise ValueError('HelixPagedAdamW8bit requires a CUDA device')
            if dev.index is None:
                dev = torch.device('cuda', torch.cuda.current_device())
            self._cuda_device = dev
            super().__init__(params, **kwargs)

        def load_state_dict(self, state_dict, move_to_device=False):
            return super().load_state_dict(state_dict, move_to_device=False)

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            if not self.initialized:
                self.check_overrides()
                self.initialized = True

            for qk, qv in list(self.name2qmap.items()):
                if isinstance(qv, torch.Tensor) and qv.device != self._cuda_device:
                    self.name2qmap[qk] = qv.to(self._cuda_device)

            last_cuda_p = None
            for gindex, group in enumerate(self.param_groups):
                for pindex, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    orig_device = p.device
                    needs_stage = orig_device.type != 'cuda'
                    if needs_stage:
                        p.data = p.data.to(self._cuda_device)
                        p.grad = p.grad.to(self._cuda_device)
                    state = self.state[p]
                    if len(state) > 0:
                        for key, val in list(state.items()):
                            if isinstance(val, torch.Tensor) and not getattr(val, 'is_paged', False) and val.device != self._cuda_device:
                                if self.is_paged and key in {'state1', 'state2'} and p.numel() >= 1e5:
                                    buff = bnb_F.get_paged(*p.shape, dtype=val.dtype, device=self._cuda_device)
                                    buff.copy_(val.to(self._cuda_device))
                                    self.page_mng.paged_tensors.append(buff)
                                    state[key] = buff
                                else:
                                    state[key] = val.to(self._cuda_device)
                    else:
                        self.init_state(group, p, gindex, pindex)

                    self.prefetch_state(p)
                    self.update_step(group, p, gindex, pindex)
                    sync_gpu(p)
                    last_cuda_p = p
                    if needs_stage:
                        p.data = p.data.to(orig_device)
                        p.grad = p.grad.to(orig_device)
            if self.is_paged and last_cuda_p is not None and last_cuda_p.is_cuda:
                sync_gpu(last_cuda_p)
            return loss

    HelixPagedAdamW8bit.__module__ = __name__
    HelixPagedAdamW8bit.__qualname__ = 'HelixPagedAdamW8bit'
    _PAGED_ADAMW8BIT_CLS = HelixPagedAdamW8bit
    return _PAGED_ADAMW8BIT_CLS


def _select_optimizer(groups, *, learning_rate: float, weight_decay: float, device: str, requested: str | None):
    import torch
    choice = (requested or 'auto').lower()
    if choice not in {'auto', 'paged_adamw_8bit', 'adamw_8bit', 'adamw', 'adamw_torch'}:
        raise ValueError(f'Unsupported Helix optimizer setting: {requested!r}')
    wants_paged = choice in {'auto', 'paged_adamw_8bit', 'adamw_8bit'}
    if wants_paged and device == 'cuda':
        try:
            opt_cls = _get_helix_paged_adamw8bit_cls()
            return opt_cls(groups, lr=learning_rate, weight_decay=weight_decay, cuda_device=device), 'PagedAdamW8bit'
        except Exception as exc:
            if choice in {'paged_adamw_8bit', 'adamw_8bit'}:
                raise RuntimeError(f'PagedAdamW8bit was requested but could not initialize: {exc}') from exc
    elif choice in {'paged_adamw_8bit', 'adamw_8bit'}:
        raise ValueError('PagedAdamW8bit requires an available CUDA device')
    return torch.optim.AdamW(groups, lr=learning_rate, weight_decay=weight_decay), 'AdamW'


def run_helix_training(config):
    _validate(config)
    import torch
    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer, get_scheduler
    from .unsloth_sft import _assistant_masked_row
    from ..adapters import resolve_training_source
    from ..dataset_fabric import resolve_manifest
    from ..helix_checkpoint import helix_target_modules, load_mapped_adapter, load_mapped_moe, save_mapped_adapter
    from ..helix_dataset_mmap import build_memmap_dataset
    from ..helix_mixtral import moe_lora_evaluate
    from ..helix_resume import _batch_digest, restore_helix_training_state, save_helix_checkpoint
    from ..helix_sft import HelixDatasetBatches, HelixPackedDatasetBatches, train_helix_batches
    from ..performance.hardware import detect_hardware
    from ..performance.planner import apply_cpu_threads, cpu_threads

    source = resolve_training_source(config)
    base = Path(source['base_model_name_or_path'] if source else config.model).resolve(strict=True)
    if not base.is_dir():
        raise ValueError('Helix requires a local MoE checkpoint directory')
    if config.backend_kind == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    device = 'cuda' if config.backend_kind in {'auto', 'cuda'} and torch.cuda.is_available() else 'cpu'
    packing = config.train.get('packing') is True or config.performance.get('packing') is True
    if packing and device != 'cuda':
        raise ValueError('Helix packed training requires CUDA FlexAttention')
    if config.performance.get('helix_prefetch', False) and device != 'cuda':
        raise ValueError('Helix layer prefetch requires CUDA')
    hardware = detect_hardware(torch)
    apply_cpu_threads(cpu_threads(hardware, config.performance), torch)
    accumulation = config.optim.get('grad_accumulation_steps', 8)
    save_every = config.train.get('save_every') or 200
    output = Path(config.output.get('adapter_path', './output')).resolve()
    if output == base or output in base.parents or base in output.parents:
        raise ValueError('Helix output must be separate from the base checkpoint directory')
    output.mkdir(parents=True, exist_ok=True)
    adapter_type = config.adapter.get('type')
    quant_bits = config.adapter.get('quant_bits') if adapter_type == 'qlora' else None
    cache = output / '.helix-nf4-cache' if (adapter_type == 'qlora' and quant_bits == 4) else None
    if source and config.output.get('resume_from_checkpoint'):
        src_path = Path(source['path'])
        nf4_marker = src_path / 'sytra-helix-nf4.json'
        quant_marker = src_path / 'sytra-helix-quant.json'
        if quant_bits == 4 and not (nf4_marker.is_file() or quant_marker.is_file()):
            raise ValueError('Resume must use the same Helix base quantization as the checkpoint')
        if quant_bits == 8 and not quant_marker.is_file():
            raise ValueError('Resume must use the same 8-bit Helix base quantization as the checkpoint')
        if quant_bits is None and (nf4_marker.is_file() or quant_marker.is_file()):
            raise ValueError('Resume must use the same Helix base quantization as the checkpoint')

    requested_steps = config.train.get('max_steps')
    requested_epochs = config.train.get('epochs')
    telemetry.emit_starting(
        config.run_id or '00000000-0000-0000-0000-000000000000',
        {'protocol_version': 1, 'op': 'train', 'backend': device, 'engine': 'helix',
         'model': str(base), 'total_steps': requested_steps or 1, 'train_mode': 'sft'},
    )
    telemetry.emit_event('stage', {'stage': 'loading_model'})
    torch.manual_seed(3407)
    if device == 'cuda':
        torch.cuda.manual_seed_all(3407)
    with telemetry.Heartbeat(lambda: {'stage': 'loading_model', 'step': 0}):
        if source:
            model, pager = load_mapped_adapter(base, source['path'], quant_cache_dir=cache, quant_bits=quant_bits)
        else:
            model_cfg = AutoConfig.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
            topology = config.adapter.get('moe_topology', 'attention')
            targets = helix_target_modules(model_cfg, topology, config.adapter.get('target_modules'))
            model, pager = load_mapped_moe(
                base,
                rank=config.adapter.get('rank', 16),
                alpha=config.adapter.get('alpha', 32),
                dropout=config.adapter.get('dropout', 0),
                use_rslora=config.adapter.get('use_rslora', False),
                target_modules=targets,
                quant_cache_dir=cache,
                quant_bits=quant_bits,
            )
        tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
    pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token is None:
        raise ValueError('Helix requires a tokenizer pad or EOS token')

    completion_only = config.algo.get('completion_only_loss', True) is not False
    path = Path(config.train_dataset or config.data.jsonl_path or '')
    if not path.is_file() or path.suffix.lower() not in {'.jsonl', '.parquet', '.json'}:
        raise ValueError('Helix requires a local JSONL, Parquet, or Fabric manifest dataset')
    files = resolve_manifest(path) if path.suffix.lower() == '.json' else str(path)
    val_path = Path(config.val_dataset) if config.val_dataset else None
    val_files = None
    if val_path is not None:
        if not val_path.is_file() or val_path.suffix.lower() not in {'.jsonl', '.parquet', '.json'}:
            raise ValueError('Helix validation requires a local JSONL, Parquet, or Fabric manifest dataset')
        val_files = resolve_manifest(val_path) if val_path.suffix.lower() == '.json' else str(val_path)
        train_paths = {Path(item).resolve() for item in (files if isinstance(files, list) else [files])}
        validation_paths = {Path(item).resolve() for item in (val_files if isinstance(val_files, list) else [val_files])}
        if train_paths & validation_paths:
            raise ValueError('Training and validation must use separate dataset files')

    telemetry.emit_event('stage', {'stage': 'preparing_dataset'})
    output.mkdir(parents=True, exist_ok=True)
    train_mmap = None
    val_mmap = None
    max_seq_len = config.train.get('max_seq_len', 2048)
    batch_size = config.train.get('batch_size', 2)
    row_mapper = _assistant_masked_row if completion_only else _causal_masked_row
    try:
        with telemetry.Heartbeat(lambda: {'stage': 'preparing_dataset', 'step': 0}):
            dataset = load_dataset(
                'parquet' if path.suffix.lower() == '.parquet' else 'json',
                data_files=files,
                split='train',
                cache_dir=str(output / '.dataset-cache'),
            )
            if not len(dataset):
                raise ValueError('Training dataset is empty')
            dataset = dataset.map(
                lambda row: row_mapper(row, tokenizer, max_seq_len),
                remove_columns=dataset.column_names,
            )
            train_mmap = build_memmap_dataset(
                dataset,
                output / '.helix-mmap-cache' / 'train',
                fingerprint=f'{path.stat().st_size}:{max_seq_len}:{completion_only}',
                force=True,
            )
            batches = (HelixPackedDatasetBatches(train_mmap, max_seq_len) if packing else
                       HelixDatasetBatches(train_mmap, batch_size, pad_token))
            val_batches = None
            val_digest = None
            if val_path is not None:
                val_dataset = load_dataset(
                    'parquet' if val_path.suffix.lower() == '.parquet' else 'json',
                    data_files=val_files,
                    split='train',
                    cache_dir=str(output / '.dataset-cache'),
                )
                if not len(val_dataset):
                    raise ValueError('Validation dataset is empty')
                val_dataset = val_dataset.map(
                    lambda row: row_mapper(row, tokenizer, max_seq_len),
                    remove_columns=val_dataset.column_names,
                )
                val_mmap = build_memmap_dataset(
                    val_dataset,
                    output / '.helix-mmap-cache' / 'val',
                    fingerprint=f'{val_path.stat().st_size}:{max_seq_len}:{completion_only}',
                    force=True,
                )
                val_batches = (HelixPackedDatasetBatches(val_mmap, max_seq_len) if packing else
                               HelixDatasetBatches(val_mmap, batch_size, pad_token))
                val_digest = _batch_digest(val_batches)

        steps_per_epoch = max(1, math.ceil(len(batches) / accumulation))
        max_steps = int(requested_steps) if requested_steps else int(requested_epochs) * steps_per_epoch

        named_trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        router = [parameter for name, parameter in named_trainable if '.gate.router_projection.lora_' in name]
        ordinary = [parameter for name, parameter in named_trainable if '.gate.router_projection.lora_' not in name]
        ratio = config.optim.get('router_lr_ratio')
        if router and ratio is None and not config.output.get('resume_from_checkpoint'):
            ratio = 0.1
            config.optim['router_lr_ratio'] = ratio
        if (ratio is None) != (not router):
            raise ValueError('Router learning-rate ratio requires identifiable router adapters and vice versa')
        auxiliary = config.optim.get('router_aux_coef') or 0.0
        if auxiliary and not router:
            raise ValueError('Router auxiliary loss requires trainable Helix router adapters')
        learning_rate = config.optim.get('learning_rate', 2e-4)
        groups = [{'params': ordinary}] if ordinary else []
        if router:
            groups.append({'params': router, 'lr': learning_rate * ratio})

        optimizer, optimizer_kind = _select_optimizer(
            groups,
            learning_rate=learning_rate,
            weight_decay=config.optim.get('weight_decay', 0),
            device=device,
            requested=config.optim.get('optimizer'),
        )
        schedule = config.optim.get('schedule', 'cosine')
        scheduler = get_scheduler(
            schedule,
            optimizer=optimizer,
            num_warmup_steps=config.optim.get('warmup_steps', 20),
            num_training_steps=max_steps,
        )
        identity: dict[str, Any] = {
            'max_steps': max_steps,
            'schedule': schedule,
            'warmup_steps': config.optim.get('warmup_steps', 20),
            'learning_rate': learning_rate,
            'weight_decay': config.optim.get('weight_decay', 0),
            'router_lr_ratio': ratio,
        }
        if requested_epochs is not None:
            identity['epochs'] = requested_epochs
        if not completion_only:
            identity['completion_only_loss'] = False
        if auxiliary:
            identity['router_aux_coef'] = auxiliary
        noise = config.train.get('neftune_noise_alpha') or 0.0
        if noise:
            identity['neftune_noise_alpha'] = noise
        if packing:
            identity['packing'] = 'flex_attention_v1'
        if val_batches is not None:
            identity['validation_digest'] = val_digest
        if quant_bits == 4:
            identity['quantization'] = 'sytra_nf4_v1'
        elif quant_bits == 8:
            identity['quantization'] = 'sytra_int8_v1'

        start = 0
        resume = config.output.get('resume_from_checkpoint')
        if resume:
            start = restore_helix_training_state(
                resume,
                model,
                optimizer,
                scheduler,
                batches,
                accumulation_steps=accumulation,
                device=device,
                identity=identity,
            )
            if start >= max_steps:
                raise ValueError('Resume checkpoint already reached max_steps')
        output.mkdir(parents=True, exist_ok=True)
        (output / 'sytra-provenance.json').write_text(
            json.dumps({
                'model': str(base),
                'engine': 'helix',
                'optimizer': optimizer_kind,
                'total_steps': max_steps,
                'adapter': config.adapter,
                'optim': config.optim,
                'train': config.train,
            }, indent=2),
            encoding='utf-8',
        )
        telemetry.emit_event('stage', {'stage': 'training'})
        last_loss = None
        last_eval_loss = None
        latest_stats: dict[str, Any] = {}

        def on_stats(stats: dict[str, Any]):
            nonlocal latest_stats
            latest_stats = stats

        def on_step(step, loss, tokens):
            nonlocal last_loss, last_eval_loss
            last_loss = loss
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == 'cuda' else 0.0
            rss_mb = _process_rss_mb()
            telemetry.emit_metric(
                step=step,
                progress=step / max_steps,
                loss=loss,
                lr=optimizer.param_groups[0]['lr'],
                tokens=tokens,
                vram_peak_mb=round(vram_mb, 2),
                rss_mb=round(rss_mb, 2),
                gpu_wait_ms=round(latest_stats.get('gpu_wait_ms', 0.0), 3),
                activation_spill_count=latest_stats.get('activation_spill_count', 0),
                optimizer=optimizer_kind,
            )
            if val_batches is not None and (step % save_every == 0 or step == max_steps):
                telemetry.emit_event('stage', {'stage': 'evaluating'})
                with telemetry.Heartbeat(lambda: {'stage': 'evaluating', 'step': step}):
                    total_loss = 0.0
                    total_tokens = 0
                    for batch in val_batches:
                        result = moe_lora_evaluate(
                            model,
                            batch.input_ids,
                            batch.attention_mask,
                            batch.labels,
                            device=device,
                            weight_pager=pager,
                            cu_seqlens=getattr(batch, 'cu_seqlens', None),
                        )
                        total_loss += result.loss * result.supervised_tokens
                        total_tokens += result.supervised_tokens
                    last_eval_loss = total_loss / total_tokens
                telemetry.emit_event('eval', {'step': step, 'loss': last_eval_loss, 'tokens': total_tokens})
                telemetry.emit_event('stage', {'stage': 'training'})
            if step % save_every == 0:
                save_helix_checkpoint(
                    output / f'checkpoint-{step}',
                    model,
                    optimizer,
                    scheduler,
                    batches,
                    step=step,
                    accumulation_steps=accumulation,
                    device=device,
                    identity=identity,
                )

        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        train_helix_batches(
            model,
            pager,
            batches,
            optimizer,
            max_steps=max_steps,
            start_step=start,
            accumulation_steps=accumulation,
            scheduler=scheduler,
            device=device,
            on_step=on_step,
            on_stats=on_stats,
            router_aux_coef=auxiliary,
            activation_memory_bytes=config.performance.get('activation_memory_mb', 512) * 1024 * 1024,
            activation_spill_dir=str(output),
            prefetch_layers=config.performance.get('helix_prefetch', False),
            prefetch_pinned_memory_bytes=config.performance.get('prefetch_pinned_memory_mb', 256) * 1024 * 1024,
            neftune_noise_alpha=noise,
            prefetch_batches=True,
        )
        telemetry.emit_event('stage', {'stage': 'saving_adapter'})
        save_mapped_adapter(model, output)
        tokenizer.save_pretrained(str(output))
        telemetry.emit_done({
            'adapter_path': str(output),
            'steps': max_steps,
            'final_loss': last_loss,
            'optimizer': optimizer_kind,
            **({'eval_loss': last_eval_loss} if last_eval_loss is not None else {}),
        })
        return 0
    finally:
        if train_mmap is not None:
            train_mmap.close()
        if val_mmap is not None:
            val_mmap.close()
