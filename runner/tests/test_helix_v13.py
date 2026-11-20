"""Verification tests for Sytra-Helix 1.3.0 multi-architecture, quant, Triton, memmap, and schedule features."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')
pytest.importorskip('peft')
pytest.importorskip('datasets')

from peft import LoraConfig, get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    DeepseekV2Config,
    DeepseekV2ForCausalLM,
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
    PreTrainedTokenizerFast,
    Qwen2MoeConfig,
    Qwen2MoeForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from sytra_runner.backends.helix_sft import _causal_masked_row, _select_optimizer, run_helix_training
from sytra_runner.config import RunConfig
from sytra_runner.helix_checkpoint import (
    SUPPORTED_MOE_TYPES,
    helix_target_modules,
    load_mapped_adapter,
    load_mapped_moe,
    save_mapped_adapter,
)
from sytra_runner.helix_dataset_mmap import MmapTokenDataset, build_memmap_dataset
from sytra_runner.helix_mixtral import moe_lora_backward, moe_lora_evaluate
from sytra_runner.helix_nf4 import FrozenNF4Linear, NF4LoRALinear, NF4_VALUES, quantize_nf4
from sytra_runner.helix_nf4_peft import NF4PeftLinear
from sytra_runner.helix_packing import packed_causal_4d_mask
from sytra_runner.helix_quant import (
    FrozenInt8Linear,
    Int8LoRALinear,
    detect_checkpoint_quantization,
    load_quantized_linear_from_sources,
    quantize_int8,
)
from sytra_runner.helix_triton_ops import (
    chunked_cross_entropy_sum,
    fused_nf4_lora_linear,
    triton_kernels_available,
)
from test_performance import config_raw


def _write_tiny_tokenizer(directory: Path) -> None:
    vocab = {'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, '<|user|>': 3, '<|assistant|>': 4, 'hello': 5, 'world': 6}
    tok = __import__('tokenizers').Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    tok.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token='[UNK]',
        pad_token='[PAD]',
        eos_token='[EOS]',
        additional_special_tokens=['<|user|>', '<|assistant|>'],
    )
    fast.chat_template = (
        "{% for m in messages %}{{ '<|' + m['role'] + '|>' }} {{ m['content'] }} {% endfor %}"
        "{% if add_generation_prompt %}{{ '<|assistant|>' }} {% endif %}"
    )
    fast.save_pretrained(str(directory))


@pytest.mark.parametrize(
    'arch,cls,cfg_kwargs',
    [
        (
            'qwen2_moe',
            Qwen2MoeForCausalLM,
            dict(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=48,
                moe_intermediate_size=24,
                shared_expert_intermediate_size=24,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_experts=3,
                num_experts_per_tok=2,
                decoder_sparse_step=1,
                max_position_embeddings=32,
                use_cache=False,
            ),
        ),
        (
            'qwen3_moe',
            Qwen3MoeForCausalLM,
            dict(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=48,
                moe_intermediate_size=24,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_experts=3,
                num_experts_per_tok=2,
                decoder_sparse_step=1,
                max_position_embeddings=32,
                use_cache=False,
            ),
        ),
        (
            'deepseek_v2',
            DeepseekV2ForCausalLM,
            dict(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=48,
                moe_intermediate_size=24,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                kv_lora_rank=8,
                q_lora_rank=16,
                qk_nope_head_dim=8,
                qk_rope_head_dim=8,
                v_head_dim=8,
                n_routed_experts=3,
                n_shared_experts=1,
                num_experts_per_tok=2,
                first_k_dense_replace=1,
                max_position_embeddings=32,
                use_cache=False,
            ),
        ),
        (
            'deepseek_v3',
            DeepseekV3ForCausalLM,
            dict(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=48,
                moe_intermediate_size=24,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                kv_lora_rank=8,
                q_lora_rank=16,
                qk_nope_head_dim=8,
                qk_rope_head_dim=8,
                v_head_dim=8,
                n_routed_experts=3,
                n_shared_experts=1,
                num_experts_per_tok=2,
                n_group=1,
                topk_group=1,
                first_k_dense_replace=1,
                max_position_embeddings=32,
                use_cache=False,
            ),
        ),
    ],
)
def test_multi_arch_moe_hybrid_and_full_load_backward_and_continuation(tmp_path, arch, cls, cfg_kwargs):
    assert arch in SUPPORTED_MOE_TYPES
    cfg = cls.config_class(**cfg_kwargs)
    torch.manual_seed(7)
    ref_model = cls(cfg).eval()
    ckpt = tmp_path / arch
    ref_model.save_pretrained(str(ckpt))
    _write_tiny_tokenizer(ckpt)

    targets = helix_target_modules(cfg, 'moe_full')
    assert 'router_projection' in targets
    model, pager = load_mapped_moe(ckpt, rank=2, alpha=4, target_modules=targets)
    if arch in {'qwen2_moe', 'deepseek_v2', 'deepseek_v3'}:
        names = [name for name, _ in model.named_parameters()]
        assert any('shared_expert' in name for name in names)

    input_ids = torch.tensor([[3, 5, 6, 4, 6]], dtype=torch.long)
    mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, -100, -100, 6, 2]], dtype=torch.long)
    res = moe_lora_backward(model, input_ids, mask, labels, device='cpu', weight_pager=pager)
    assert res.supervised_tokens == 2
    assert torch.isfinite(torch.tensor(res.loss))

    adapter_dir = tmp_path / f'{arch}-adapter'
    save_mapped_adapter(model, adapter_dir)
    reloaded, _ = load_mapped_adapter(ckpt, adapter_dir)
    saved_weights = load_file(str(adapter_dir / 'adapter_model.safetensors'))
    reloaded_state = get_peft_model_state_dict(reloaded, save_embedding_layers=False)
    assert set(saved_weights) == set(reloaded_state)
    for key, expected_tensor in saved_weights.items():
        torch.testing.assert_close(reloaded_state[key], expected_tensor, atol=0, rtol=0)
    orig_trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    reloaded_trainable = {name for name, p in reloaded.named_parameters() if p.requires_grad}
    assert orig_trainable and reloaded_trainable == orig_trainable


def test_deepseek_target_modules_q_lora_rank_and_qwen2_nf4_bias(tmp_path):
    for cfg_cls in (DeepseekV2Config, DeepseekV3Config):
        common = dict(
            vocab_size=32, hidden_size=32, intermediate_size=48, moe_intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, kv_lora_rank=8,
            qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=8, n_routed_experts=2,
            n_shared_experts=1, num_experts_per_tok=1, first_k_dense_replace=1, max_position_embeddings=32,
        )
        t_with_q_lora = helix_target_modules(cfg_cls(**common, q_lora_rank=16), 'attention')
        assert 'q_a_proj' in t_with_q_lora and 'q_b_proj' in t_with_q_lora and 'q_proj' not in t_with_q_lora
        t_no_q_lora = helix_target_modules(cfg_cls(**common, q_lora_rank=None), 'attention')
        assert 'q_proj' in t_no_q_lora and 'q_a_proj' not in t_no_q_lora and 'q_b_proj' not in t_no_q_lora

    # Verify Qwen2Moe preserves .bias through cached 4-bit NF4 mapping and pager restore.
    qcfg = Qwen2MoeConfig(
        vocab_size=32, hidden_size=32, intermediate_size=48, moe_intermediate_size=24,
        shared_expert_intermediate_size=24, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, num_experts=2, num_experts_per_tok=1, decoder_sparse_step=1,
        max_position_embeddings=32, use_cache=False,
    )
    torch.manual_seed(13)
    ref = Qwen2MoeForCausalLM(qcfg).eval()
    ckpt = tmp_path / 'qwen2_nf4'
    ref.save_pretrained(str(ckpt))
    model, pager = load_mapped_moe(ckpt, rank=2, alpha=4, quant_cache_dir=tmp_path / 'nf4_cache', quant_bits=4)
    q_proj = model.get_base_model().model.layers[0].self_attn.q_proj
    assert isinstance(q_proj.base_layer, FrozenNF4Linear)
    assert q_proj.base_layer.bias is not None and not q_proj.base_layer.bias.requires_grad
    torch.testing.assert_close(q_proj.base_layer.bias, ref.model.layers[0].self_attn.q_proj.bias, atol=0, rtol=0)
    ids = torch.tensor([[3, 5, 6, 4]], dtype=torch.long)
    res = moe_lora_backward(model, ids, torch.ones_like(ids), torch.tensor([[-100, 5, 6, 4]], dtype=torch.long),
                            device='cpu', weight_pager=pager)
    assert torch.isfinite(torch.tensor(res.loss))
    pager.restore(model)


def test_frozen_int8_and_bnb_prequantized_detection_and_parity(tmp_path):
    torch.manual_seed(11)
    weight = torch.randn(16, 32, dtype=torch.float32)
    linear = FrozenInt8Linear.from_float(weight)
    lora = Int8LoRALinear(linear, rank=4, alpha=8)
    x = torch.randn(2, 5, 32, dtype=torch.float32, requires_grad=True)
    y = lora(x)
    y.sum().backward()
    assert torch.isfinite(y).all()
    assert lora.lora_A.weight.grad is not None and lora.lora_B.weight.grad is not None

    q, scales = quantize_int8(weight)
    assert q.dtype == torch.int8 and scales.shape == (16,)
    reconstructed = q.float() * scales[:, None]
    torch.testing.assert_close(reconstructed, weight, atol=0.08, rtol=0.05)

    bnb_linear = FrozenInt8Linear.from_bnb(q, scales * 127.0)
    torch.testing.assert_close(bnb_linear(x.detach()), linear(x.detach()), atol=1e-6, rtol=1e-6)

    bias = torch.randn(16, dtype=torch.float32)
    biased_i8 = FrozenInt8Linear.from_float(weight, bias=bias)
    biased_i8_bnb = FrozenInt8Linear.from_bnb(q, scales * 127.0, bias=bias)
    ref_i8 = torch.nn.functional.linear(x.detach(), reconstructed, bias)
    torch.testing.assert_close(biased_i8(x.detach()), ref_i8, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(biased_i8_bnb(x.detach()), ref_i8, atol=1e-5, rtol=1e-5)

    biased_nf4 = FrozenNF4Linear.from_float(weight, block_size=8, bias=bias)
    nf4_codes = torch.tensor(NF4_VALUES, dtype=torch.float32)
    idx = torch.arange(weight.numel(), dtype=torch.long)
    encoded = biased_nf4.packed_weight[idx // 2]
    nibbles = torch.where(idx % 2 == 0, encoded & 15, encoded >> 4).long()
    nf4_weight = (nf4_codes[nibbles] * biased_nf4.weight_scales[idx // 8]).view_as(weight)
    ref_nf4 = torch.nn.functional.linear(x.detach(), nf4_weight, bias)
    torch.testing.assert_close(biased_nf4(x.detach()), ref_nf4, atol=1e-5, rtol=1e-5)

    # Verify load_quantized_linear_from_sources preserves .bias for both 8-bit and 4-bit BNB checkpoints.
    bnb_path = tmp_path / 'bnb.safetensors'
    qstate = torch.tensor(list(json.dumps({'shape': [16, 32], 'blocksize': 8, 'quant_type': 'nf4'}).encode('utf-8')), dtype=torch.uint8)
    save_file(
        {
            'i8.weight': q,
            'i8.SCB': (scales * 127.0).contiguous(),
            'i8.bias': bias,
            'nf4.weight': biased_nf4.packed_weight,
            'nf4.weight.absmax': biased_nf4.weight_scales,
            'nf4.weight.quant_state.bitsandbytes__nf4': qstate,
            'nf4.bias': bias.clone(),
        },
        str(bnb_path),
    )
    sources = {
        k: (bnb_path, k)
        for k in ('i8.weight', 'i8.SCB', 'i8.bias', 'nf4.weight', 'nf4.weight.absmax', 'nf4.weight.quant_state.bitsandbytes__nf4', 'nf4.bias')
    }
    loaded_i8 = load_quantized_linear_from_sources(sources, 'i8')
    assert isinstance(loaded_i8, FrozenInt8Linear) and loaded_i8.bias is not None
    torch.testing.assert_close(loaded_i8(x.detach()), ref_i8, atol=1e-5, rtol=1e-5)
    loaded_nf4 = load_quantized_linear_from_sources(sources, 'nf4')
    assert isinstance(loaded_nf4, FrozenNF4Linear) and loaded_nf4.bias is not None
    torch.testing.assert_close(loaded_nf4.bias, bias, atol=0, rtol=0)

    meta_4 = detect_checkpoint_quantization({'quantization_config': {'quant_method': 'bitsandbytes', 'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_use_double_quant': True}})
    meta_8 = detect_checkpoint_quantization({'quantization_config': {'quant_method': 'bitsandbytes', 'load_in_8bit': True}})
    assert meta_4 is not None and meta_4['bits'] == 4 and meta_4['double_quant'] is True
    assert meta_8 is not None and meta_8['bits'] == 8


def test_memmap_dataset_roundtrip_and_causal_loss_row(tmp_path):
    _write_tiny_tokenizer(tmp_path / 'tok')
    tok = transformers.AutoTokenizer.from_pretrained(str(tmp_path / 'tok'), local_files_only=True)
    rows = [
        _causal_masked_row({'prompt': 'hello', 'completion': 'world'}, tok, max_length=32),
        {'input_ids': [3, 5, 6], 'assistant_masks': [0, 0, 1]},
    ]
    assert rows[0]['assistant_masks'][0] == 0
    assert all(m == 1 for m in rows[0]['assistant_masks'][1:])

    cache = tmp_path / 'mmap'
    ds = build_memmap_dataset(rows, cache, fingerprint='fp-1', force=True)
    assert len(ds) == 2
    reloaded = MmapTokenDataset(cache)
    assert len(reloaded) == 2
    assert reloaded[0]['input_ids'].tolist() == rows[0]['input_ids']
    assert reloaded[1]['assistant_masks'].tolist() == rows[1]['assistant_masks']
    assert reloaded.sequence_length(0) == len(rows[0]['input_ids'])
    reloaded.close()
    ds.close()


def test_packed_causal_4d_mask_matches_block_isolation():
    cu = torch.tensor([0, 3, 5], dtype=torch.long)
    pos, mask = packed_causal_4d_mask(cu, dtype=torch.float32, device='cpu')
    assert pos.tolist() == [[0, 1, 2, 0, 1]]
    assert mask.shape == (1, 1, 5, 5)
    # Token 3 (doc 2 start) cannot attend to tokens 0..2 (doc 1)
    assert (mask[0, 0, 3, :3] < -1e20).all()
    assert mask[0, 0, 3, 3] == 0.0
    assert mask[0, 0, 4, 3] == 0.0 and mask[0, 0, 4, 4] == 0.0


def test_select_optimizer_cpu_fallback_and_cuda_rejects_cpu_device():
    param = torch.nn.Parameter(torch.zeros(4, 4))
    opt, kind = _select_optimizer([{'params': [param]}], learning_rate=1e-4, weight_decay=0.0, device='cpu', requested='auto')
    assert kind == 'AdamW' and isinstance(opt, torch.optim.AdamW)
    with pytest.raises(ValueError, match='CUDA'):
        _select_optimizer([{'params': [param]}], learning_rate=1e-4, weight_decay=0.0, device='cpu', requested='paged_adamw_8bit')


def test_epochs_completion_only_false_and_base_weight_immutability(tmp_path):
    cfg = Qwen2MoeConfig(
        vocab_size=32, hidden_size=32, intermediate_size=48, moe_intermediate_size=24,
        shared_expert_intermediate_size=24, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, num_experts=2, num_experts_per_tok=1, decoder_sparse_step=1,
        max_position_embeddings=32, use_cache=False,
    )
    base = tmp_path / 'base'
    Qwen2MoeForCausalLM(cfg).save_pretrained(str(base))
    _write_tiny_tokenizer(base)
    before_hashes = {p.name: load_file(str(p)) for p in base.glob('*.safetensors')}

    data = tmp_path / 'train.jsonl'
    data.write_text(json.dumps({'prompt': 'hello', 'completion': 'world'}) + '\n', encoding='utf-8')
    raw = config_raw(tmp_path)
    raw.update(model=str(base), train_dataset=str(data), backend={'kind': 'cpu'}, performance={'training_engine': 'helix'})
    raw['adapter'].update(type='qlora', rank=2, alpha=4, dropout=0, quant_bits=8, moe_topology='moe_hybrid')
    raw['train'].update(max_steps=None, epochs=2, batch_size=1, save_every=1, max_seq_len=32, packing=False)
    raw['optim'].update(grad_accumulation_steps=1, warmup_steps=0)
    raw['algo'] = {'completion_only_loss': False}
    assert run_helix_training(RunConfig.from_dict(raw)) == 0

    output = Path(raw['output']['adapter_path'])
    assert (output / 'sytra-helix-quant.json').is_file()
    after_hashes = {p.name: load_file(str(p)) for p in base.glob('*.safetensors')}
    for fname, tensors in before_hashes.items():
        for k, t in tensors.items():
            torch.testing.assert_close(t, after_hashes[fname][k])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_triton_fused_nf4_lora_and_chunked_ce_parity():
    if not triton_kernels_available('cuda'):
        pytest.skip('Triton CUDA kernels unavailable')
    torch.manual_seed(3)
    w_cpu = torch.randn(32, 64, dtype=torch.float16)
    packed, scales = quantize_nf4(w_cpu, block_size=32)
    packed, scales = packed.cuda(), scales.cuda()
    codebook = torch.tensor(NF4_VALUES, dtype=torch.float32, device='cuda')
    x = torch.randn(4, 64, device='cuda', dtype=torch.float16, requires_grad=True)
    a = torch.randn(4, 64, device='cuda', dtype=torch.float16, requires_grad=True)
    b = torch.randn(32, 4, device='cuda', dtype=torch.float16, requires_grad=True)
    out = fused_nf4_lora_linear(
        x, a, b, packed, scales, codebook, 32, 64, 32,
        low_first=True, lora_scale=2.0, dropout_p=0.0, training=True,
    )
    assert out.shape == (4, 32)
    out.sum().backward()
    assert x.grad is not None and a.grad is not None and b.grad is not None

    bias = torch.randn(32, device='cuda', dtype=torch.float16)
    nf4_base = FrozenNF4Linear(packed, scales, 64, 32, block_size=32, codebook=codebook, bias=bias)
    nf4_lora = NF4LoRALinear(nf4_base, rank=4, alpha=8.0).cuda()
    torch.nn.init.normal_(nf4_lora.lora_B.weight, std=0.05)
    x_bias = x.detach().clone().requires_grad_(True)
    y_fused = nf4_lora(x_bias)
    idx = torch.arange(32 * 64, device='cuda', dtype=torch.long)
    enc = packed[idx // 2]
    nib = torch.where(idx % 2 == 0, enc & 15, enc >> 4).long()
    w_deq = (codebook[nib] * scales[idx // 32]).view(32, 64).to(torch.float16)
    y_ref = (
        torch.nn.functional.linear(x_bias, w_deq, bias)
        + torch.nn.functional.linear(
            torch.nn.functional.linear(x_bias.float(), nf4_lora.lora_A.weight.float()),
            nf4_lora.lora_B.weight.float(),
        ).to(torch.float16) * nf4_lora.scaling
    )
    torch.testing.assert_close(y_fused, y_ref, atol=5e-2, rtol=2e-2)

    peft_cfg = LoraConfig(r=4, lora_alpha=8)
    peft_nf4 = NF4PeftLinear(nf4_base, 'default', peft_cfg, r=4, lora_alpha=8).cuda()
    with torch.no_grad():
        peft_nf4.lora_A['default'].weight.copy_(nf4_lora.lora_A.weight)
        peft_nf4.lora_B['default'].weight.copy_(nf4_lora.lora_B.weight)
    torch.testing.assert_close(peft_nf4(x_bias), y_ref, atol=5e-2, rtol=2e-2)
    peft_nf4.enable_adapters(False)
    torch.testing.assert_close(peft_nf4(x_bias), torch.nn.functional.linear(x_bias, w_deq, bias), atol=5e-2, rtol=2e-2)

    hidden = torch.randn(6, 32, device='cuda', dtype=torch.float32, requires_grad=True)
    lm_head = torch.randn(256, 32, device='cuda', dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([1, 5, -100, 12, 99, 2], device='cuda', dtype=torch.long)
    ce_sum = chunked_cross_entropy_sum(hidden @ lm_head.T, labels, ignore_index=-100)
    ref_sum = torch.nn.functional.cross_entropy(hidden @ lm_head.T, labels, ignore_index=-100, reduction='sum')
    torch.testing.assert_close(ce_sum, ref_sum, atol=1e-4, rtol=1e-4)
