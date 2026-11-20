"""Exercise local checkpoint loading through a full Mixtral backward pass."""
import copy
import json

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
pytest.importorskip('peft')

from sytra_runner.helix_checkpoint import _checkpoint_sources, load_mapped_adapter, load_mapped_mixtral, save_mapped_adapter
from sytra_runner.helix_mixtral import mixtral_lora_backward
from test_helix_mixtral import make_model


@pytest.mark.parametrize('sharded', [False, True])
def test_load_mapped_mixtral_matches_autograd(tmp_path, sharded):
    model = make_model()
    # Save an ordinary base checkpoint, without the test-only adapter wrappers.
    for layer in model.model.layers:
        layer.self_attn.q_proj = layer.self_attn.q_proj.base_layer
    model.save_pretrained(str(tmp_path), max_shard_size='3KB' if sharded else '1GB')
    loaded, pager = load_mapped_mixtral(tmp_path, rank=2, alpha=4)
    assert pager.mapped_parameters > 0
    reference = copy.deepcopy(loaded)
    ids = torch.tensor([[1, 5, 6, 7], [1, 9, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    labels = torch.tensor([[-100, 5, 6, 7], [-100, 9, -100, -100]])
    model.eval()
    loaded.eval()
    with torch.no_grad():
        original_logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        mapped_logits = loaded(input_ids=ids, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(mapped_logits, original_logits, atol=1e-6, rtol=1e-4)
    pager.restore(loaded)
    loaded.train()
    expected = reference(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
    expected.backward()
    result = mixtral_lora_backward(loaded, ids, mask, labels, weight_pager=pager)
    assert result.loss == pytest.approx(expected.item(), abs=1e-6)
    expected_params = dict(reference.named_parameters())
    for name, parameter in loaded.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, expected_params[name].grad, atol=1e-6, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_only_selected_expert_moves_to_cuda_and_is_released(tmp_path):
    model = make_model()
    for layer in model.model.layers:
        layer.self_attn.q_proj = layer.self_attn.q_proj.base_layer
    model.save_pretrained(str(tmp_path))
    loaded, pager = load_mapped_mixtral(tmp_path, rank=2, alpha=4)
    experts = loaded.get_base_model().model.layers[0].mlp.experts
    input_data = torch.randn(3, 16, device='cuda', requires_grad=True)
    output = experts(input_data, torch.zeros((3, 1), device='cuda', dtype=torch.long),
                     torch.ones((3, 1), device='cuda'))
    assert next(experts[0].parameters()).device.type == 'cuda'
    assert next(experts[1].parameters()).device.type == 'cpu'
    output.sum().backward()
    assert input_data.grad is not None
    pager.restore(experts)
    assert all(parameter.device.type == 'cpu' for parameter in experts.parameters())
    assert not experts.loaded


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_windowed_cuda_trains_selected_expert_adapters(tmp_path):
    model = make_model(top_k=2)
    for layer in model.model.layers:
        layer.self_attn.q_proj = layer.self_attn.q_proj.base_layer
    model.save_pretrained(str(tmp_path))
    loaded, pager = load_mapped_mixtral(tmp_path, rank=2, alpha=4,
                                        target_modules=('q_proj', 'w1', 'w2', 'w3', 'router_projection'))
    reference = copy.deepcopy(loaded)
    ids = torch.tensor([[1, 5, 6, 7]])
    mask = torch.ones_like(ids)
    labels = torch.tensor([[-100, 5, 6, 7]])
    expected = reference(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
    expected.backward()
    result = mixtral_lora_backward(loaded, ids, mask, labels, device='cuda', weight_pager=pager)
    assert result.loss == pytest.approx(expected.item(), abs=1e-5)
    expected_params = dict(reference.named_parameters())
    expert_gradients = 0
    router_gradients = 0
    for name, parameter in loaded.named_parameters():
        if not parameter.requires_grad:
            continue
        expected_gradient = expected_params[name].grad
        assert (parameter.grad is None) == (expected_gradient is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, expected_gradient, atol=2e-5, rtol=1e-3)
            expert_gradients += int('.experts.' in name)
            if '.gate.router_projection.lora_B.' in name:
                router_gradients += int(parameter.grad.abs().sum() > 0)
        assert parameter.device.type == 'cpu'
    assert expert_gradients > 0
    assert router_gradients > 0


def test_checkpoint_index_rejects_outside_shards(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + '-outside.safetensors')
    outside.write_bytes(b'not a checkpoint')
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': '../' + outside.name}}))
    try:
        with pytest.raises(ValueError, match='inside the checkpoint'):
            _checkpoint_sources(tmp_path)
    finally:
        outside.unlink()


@pytest.mark.parametrize('targets', [('q_proj', 'v_proj'), ('q_proj', 'w1', 'w2', 'w3', 'router_projection')])
def test_saved_adapter_reloads_exactly(tmp_path, targets):
    base = tmp_path / 'base'
    output = tmp_path / 'adapter'
    model = make_model(top_k=2)
    for layer in model.model.layers:
        layer.self_attn.q_proj = layer.self_attn.q_proj.base_layer
    model.save_pretrained(str(base))
    trained, pager = load_mapped_mixtral(base, rank=2, alpha=4, target_modules=targets)
    ids = torch.tensor([[1, 5, 6, 7]])
    labels = torch.tensor([[-100, 5, 6, 7]])
    optimizer = torch.optim.AdamW([p for p in trained.parameters() if p.requires_grad], lr=.01)
    mixtral_lora_backward(trained, ids, torch.ones_like(ids), labels, weight_pager=pager)
    optimizer.step()
    trained.save_pretrained(str(output), safe_serialization=True, save_embedding_layers=False)
    loaded, loaded_pager = load_mapped_adapter(base, output)
    original = {n: p for n, p in trained.named_parameters() if p.requires_grad}
    restored = {n: p for n, p in loaded.named_parameters() if p.requires_grad}
    assert set(original) == set(restored)
    for name in original:
        torch.testing.assert_close(original[name], restored[name], atol=0, rtol=0)
    reference = mixtral_lora_backward(trained, ids, torch.ones_like(ids), labels, weight_pager=pager)
    actual = mixtral_lora_backward(loaded, ids, torch.ones_like(ids), labels, weight_pager=loaded_pager)
    assert actual.loss == pytest.approx(reference.loss, abs=1e-7)


def test_unquantized_loader_restores_peft_dispatchers(tmp_path, monkeypatch):
    import peft
    import peft.tuners.lora.model as lora_model
    model = make_model(top_k=2)
    for layer in model.model.layers:
        layer.self_attn.q_proj = layer.self_attn.q_proj.base_layer
    model.save_pretrained(str(tmp_path))
    before = (lora_model.is_bnb_available, lora_model.is_bnb_4bit_available)
    def fail(*args, **kwargs):
        assert not lora_model.is_bnb_available()
        assert not lora_model.is_bnb_4bit_available()
        raise RuntimeError('adapter failure')
    monkeypatch.setattr(peft, 'get_peft_model', fail)
    with pytest.raises(RuntimeError, match='adapter failure'):
        load_mapped_mixtral(tmp_path)
    assert (lora_model.is_bnb_available, lora_model.is_bnb_4bit_available) == before


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('gradient_scale', [1.0, .4])
def test_windowed_router_balance_matches_full_autograd(tmp_path, device, gradient_scale):
    from sytra_runner.performance.moe_balance import balance_loss
    base = make_model(top_k=2, experts=3, stochastic=True)
    for layer in base.model.layers:
        layer.self_attn.q_proj = layer.self_attn.q_proj.base_layer
    base.save_pretrained(str(tmp_path))
    model, pager = load_mapped_mixtral(tmp_path, rank=2, alpha=4, dropout=.15,
                                       target_modules=('q_proj', 'w1', 'w2', 'w3', 'router_projection'))
    reference = copy.deepcopy(model)
    ids = torch.tensor([[1, 5, 6, 7], [1, 9, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    labels = torch.tensor([[-100, 5, 6, 7], [-100, 9, -100, -100]])
    if device == 'cuda':
        reference.cuda()
    captured = []
    handles = [layer.mlp.gate.router_projection.register_forward_hook(
        lambda module, inputs, output: captured.append(output)) for layer in reference.get_base_model().model.layers]
    try:
        torch.manual_seed(314)
        if device == 'cuda':
            torch.cuda.manual_seed_all(314)
        baseline = reference(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device), use_cache=False).loss
        auxiliary = balance_loss(captured, mask.to(device), num_experts=3, top_k=2)
        router = reference.get_base_model().model.layers[0].mlp.gate.router_projection.lora_B['default'].weight
        assert torch.autograd.grad(auxiliary, router, retain_graph=True)[0].abs().sum() > 0
        objective = baseline + .04 * auxiliary
        (objective * gradient_scale).backward()
    finally:
        for handle in handles:
            handle.remove()
    torch.manual_seed(314)
    if device == 'cuda':
        torch.cuda.manual_seed_all(314)
    result = mixtral_lora_backward(model, ids, mask, labels, device=device, weight_pager=pager,
                                    router_aux_coef=.04, gradient_scale=gradient_scale)
    assert result.loss == pytest.approx(objective.item(), abs=1e-6)
    expected = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert (parameter.grad is None) == (expected[name].grad is None)
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, expected[name].grad.cpu(), atol=2e-6, rtol=1e-4)
    assert not model.get_base_model().model.layers[0].mlp.gate.router_projection._forward_hooks


def test_router_balance_removes_hooks_on_failure(tmp_path):
    base = make_model(top_k=2, experts=3)
    for layer in base.model.layers:
        layer.self_attn.q_proj = layer.self_attn.q_proj.base_layer
    base.save_pretrained(str(tmp_path))
    model, pager = load_mapped_mixtral(tmp_path, rank=2, alpha=4, target_modules=('q_proj', 'router_projection'))
    ids = torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match='No supervised tokens'):
        mixtral_lora_backward(model, ids, torch.ones_like(ids), torch.full_like(ids, -100),
                              weight_pager=pager, router_aux_coef=.04)
    assert all(not layer.mlp.gate.router_projection._forward_hooks
               for layer in model.get_base_model().model.layers)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_nf4_mapped_mixtral_windowed_matches_full_autograd(tmp_path, device):
    from sytra_runner.helix_nf4 import FrozenNF4Linear
    base_dir = tmp_path / 'base'
    cache = tmp_path / 'nf4-cache'
    base = make_model(top_k=2, experts=3)
    for layer in base.model.layers:
        layer.self_attn.q_proj = layer.self_attn.q_proj.base_layer
    base.save_pretrained(str(base_dir))
    model, pager = load_mapped_mixtral(base_dir, rank=2, alpha=4,
        target_modules=('q_proj', 'w1', 'w2', 'w3', 'router_projection'), quant_cache_dir=cache)
    assert sum(isinstance(module, FrozenNF4Linear) for module in model.modules()) == 27
    assert len(list(cache.glob('*.safetensors'))) > 0
    reference = copy.deepcopy(model)
    if device == 'cuda':
        reference.cuda()
    ids = torch.tensor([[1, 5, 6, 7], [1, 9, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    labels = torch.tensor([[-100, 5, 6, 7], [-100, 9, -100, -100]])
    expected = reference(input_ids=ids.to(device), attention_mask=mask.to(device),
                         labels=labels.to(device), use_cache=False).loss
    expected.backward()
    actual = mixtral_lora_backward(model, ids, mask, labels, device=device, weight_pager=pager,
                                   prefetch_layers=device == 'cuda')
    assert actual.loss == pytest.approx(expected.item(), abs=1e-6)
    expected_parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            reference_gradient = expected_parameters[name].grad
            assert (parameter.grad is None) == (reference_gradient is None)
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, reference_gradient.cpu(), atol=2e-6, rtol=1e-4)
    assert all(module.packed_weight.device.type == 'cpu' for module in model.modules() if isinstance(module, FrozenNF4Linear))


def test_nf4_adapter_save_and_reload(tmp_path):
    base = tmp_path / 'base'
    adapter = tmp_path / 'adapter'
    original = make_model(top_k=2, experts=3)
    for layer in original.model.layers:
        layer.self_attn.q_proj = layer.self_attn.q_proj.base_layer
    original.save_pretrained(str(base))
    model, pager = load_mapped_mixtral(base, rank=2, alpha=4,
        target_modules=('q_proj', 'w1', 'w2', 'w3', 'router_projection'), quant_cache_dir=tmp_path / 'cache1')
    ids = torch.tensor([[1, 5, 6, 7]])
    labels = torch.tensor([[-100, 5, 6, 7]])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    mixtral_lora_backward(model, ids, torch.ones_like(ids), labels, weight_pager=pager, router_aux_coef=.02)
    optimizer.step()
    save_mapped_adapter(model, adapter)
    assert (adapter / 'sytra-helix-nf4.json').is_file()
    with pytest.raises(ValueError, match='requires Helix NF4'):
        load_mapped_adapter(base, adapter)
    loaded, loaded_pager = load_mapped_adapter(base, adapter, quant_cache_dir=tmp_path / 'cache2')
    expected = {name: p for name, p in model.named_parameters() if p.requires_grad}
    actual = {name: p for name, p in loaded.named_parameters() if p.requires_grad}
    assert set(actual) == set(expected)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    expected_loss = mixtral_lora_backward(model, ids, torch.ones_like(ids), labels, weight_pager=pager, router_aux_coef=.02).loss
    actual_loss = mixtral_lora_backward(loaded, ids, torch.ones_like(ids), labels, weight_pager=loaded_pager, router_aux_coef=.02).loss
    assert actual_loss == pytest.approx(expected_loss, abs=1e-7)
    marker_path = adapter / 'sytra-helix-nf4.json'
    marker = json.loads(marker_path.read_text())
    marker['source_digest'] = '0' * 64
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='source weights changed'):
        load_mapped_adapter(base, adapter, quant_cache_dir=tmp_path / 'cache2')


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_resume_matches_uninterrupted_training(tmp_path, device):
    from sytra_runner.helix_resume import save_helix_checkpoint, restore_helix_training_state
    from sytra_runner.helix_sft import HelixBatch, train_helix_batches
    base = tmp_path / 'base'
    model = make_model(top_k=2)
    for layer in model.model.layers:
        layer.self_attn.q_proj = layer.self_attn.q_proj.base_layer
    model.save_pretrained(str(base))
    batches = [HelixBatch(torch.tensor([[1, 3 + i, 6, 7]]), torch.ones(1, 4, dtype=torch.long),
                          torch.tensor([[-100, 3 + i, 6, 7]])) for i in range(3)]

    def create():
        torch.manual_seed(111)
        model, pager = load_mapped_mixtral(base, rank=2, alpha=4, dropout=.2,
                                           target_modules=('q_proj', 'w1', 'w2', 'w3', 'router_projection'))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .9 ** step)
        return model, pager, optimizer, scheduler

    baseline, baseline_pager, baseline_optimizer, baseline_scheduler = create()
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(888)
    train_helix_batches(baseline, baseline_pager, batches, baseline_optimizer, max_steps=3,
                        accumulation_steps=2, scheduler=baseline_scheduler, device=device)
    interrupted, pager, optimizer, scheduler = create()
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(888)
    checkpoint = tmp_path / 'checkpoint-1'
    def save(step, loss, tokens):
        save_helix_checkpoint(checkpoint, interrupted, optimizer, scheduler, batches, step=step,
                              accumulation_steps=2, device=device, identity={'schedule_steps': 3})
    train_helix_batches(interrupted, pager, batches, optimizer, max_steps=1, accumulation_steps=2,
                        scheduler=scheduler, device=device, on_step=save)
    resumed, resumed_pager = load_mapped_adapter(base, checkpoint)
    resumed_optimizer = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=.01)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda step: .9 ** step)
    step = restore_helix_training_state(checkpoint, resumed, resumed_optimizer, resumed_scheduler, batches,
                                        accumulation_steps=2, device=device, identity={'schedule_steps': 3})
    assert step == 1
    train_helix_batches(resumed, resumed_pager, batches, resumed_optimizer, max_steps=3, start_step=step,
                        accumulation_steps=2, scheduler=resumed_scheduler, device=device)
    expected = dict(baseline.named_parameters())
    for name, parameter in resumed.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter, expected[name], atol=1e-7, rtol=1e-6)
    assert resumed_scheduler.state_dict() == baseline_scheduler.state_dict()
    with pytest.raises(ValueError, match='dataset'):
        restore_helix_training_state(checkpoint, resumed, resumed_optimizer, resumed_scheduler, batches[::-1],
                                    accumulation_steps=2, device=device, identity={'schedule_steps': 3})
