"""Compare the windowed adapter against a real tiny Transformers Mixtral."""
import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM
from sytra_runner.helix_mixtral import mixtral_lora_backward, mixtral_lora_evaluate


class LoRA(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base_layer = base
        self.lora_A = torch.nn.Linear(base.in_features, 2, bias=False)
        self.lora_B = torch.nn.Linear(2, base.out_features, bias=False)

    def forward(self, value):
        return self.base_layer(value) + self.lora_B(self.lora_A(value))


def make_model(attention="eager", sliding_window=None, stochastic=False, top_k=1, experts=2):
    config = MixtralConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
                           num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                           num_local_experts=experts, num_experts_per_tok=top_k, max_position_embeddings=64,
                           use_cache=False, output_router_logits=False, attention_dropout=0.1 if stochastic else 0.0,
                           router_jitter_noise=0.05 if stochastic else 0.0, sliding_window=sliding_window)
    config._attn_implementation = attention
    model = MixtralForCausalLM(config)
    model.requires_grad_(False)
    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRA(layer.self_attn.q_proj)
    model.train()
    return model


@pytest.mark.parametrize("attention,sliding_window,stochastic", [("eager", None, False), ("sdpa", None, False), ("eager", 3, True)])
def test_mixtral_windowed_gradients_match_full_model(attention, sliding_window, stochastic):
    torch.manual_seed(42)
    model = make_model(attention, sliding_window, stochastic)
    windowed = copy.deepcopy(model)
    ids = torch.tensor([[1, 5, 6, 7, 2], [1, 9, 4, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    labels = torch.tensor([[-100, -100, 6, 7, 2], [-100, 9, 4, -100, -100]])
    torch.manual_seed(777)
    baseline = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
    baseline.backward()
    torch.manual_seed(777)
    result = mixtral_lora_backward(windowed, ids, mask, labels, loss_chunk_tokens=2)
    assert result.loss == pytest.approx(baseline.item(), abs=1e-6)
    for reference, actual in zip(model.model.layers, windowed.model.layers):
        torch.testing.assert_close(actual.self_attn.q_proj.lora_A.weight.grad,
                                   reference.self_attn.q_proj.lora_A.weight.grad, atol=1e-6, rtol=1e-4)
        torch.testing.assert_close(actual.self_attn.q_proj.lora_B.weight.grad,
                                   reference.self_attn.q_proj.lora_B.weight.grad, atol=1e-6, rtol=1e-4)
        assert actual.self_attn.q_proj.base_layer.weight.grad is None


def test_mixtral_rejects_router_auxiliary_loss():
    model = make_model()
    model.config.output_router_logits = True
    ids = torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match="auxiliary"):
        mixtral_lora_backward(model, ids, torch.ones_like(ids), ids)


def test_mixtral_rejects_unmasked_padding():
    model = make_model()
    ids = torch.tensor([[1, 2, 0]])
    with pytest.raises(ValueError, match="Padding"):
        mixtral_lora_backward(model, ids, torch.tensor([[1, 1, 0]]), ids)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_windowed_evaluation_matches_full_model_and_preserves_state(device):
    torch.manual_seed(26)
    model = make_model(stochastic=True)
    reference = copy.deepcopy(model).eval()
    model.model.layers[0].self_attn.q_proj.eval()
    flags = [module.training for module in model.modules()]
    ids = torch.tensor([[1, 5, 6, 7], [1, 9, 4, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    labels = torch.tensor([[-100, -100, 6, 7], [-100, 9, 4, -100]])
    with torch.no_grad():
        expected = reference(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.full_like(parameter, 0.25)
    before = torch.random.get_rng_state().clone()
    result = mixtral_lora_evaluate(model, ids, mask, labels, device=device, loss_chunk_tokens=2)
    assert result.loss == pytest.approx(expected.item(), abs=1e-5)
    assert result.supervised_tokens == 4
    assert torch.equal(before, torch.random.get_rng_state())
    assert [module.training for module in model.modules()] == flags
    for parameter in model.parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 0.25))
    with pytest.raises(ValueError, match='Padding'):
        mixtral_lora_evaluate(model, ids, mask, ids, device=device)
    assert [module.training for module in model.modules()] == flags
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_neftune_matches_full_model_loss_and_gradients(device):
    torch.manual_seed(42)
    model = make_model()
    windowed = copy.deepcopy(model)
    ids = torch.tensor([[1, 5, 6, 7], [1, 9, 4, 2]])
    labels = torch.tensor([[-100, -100, 6, 7], [-100, 9, 4, 2]])
    alpha = 5.0

    def add_noise(_module, _inputs, output):
        magnitude = alpha / (output.shape[1] * output.shape[2]) ** .5
        return output + torch.empty_like(output).uniform_(-magnitude, magnitude)

    handle = model.model.embed_tokens.register_forward_hook(add_noise)
    torch.manual_seed(123)
    reference = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss
    reference.backward()
    handle.remove()
    torch.manual_seed(123)
    actual = mixtral_lora_backward(windowed, ids, torch.ones_like(ids), labels,
                                   device=device, neftune_noise_alpha=alpha)
    assert actual.loss == pytest.approx(reference.item(), abs=1e-5)
    for expected, result in zip(model.parameters(), windowed.parameters()):
        if expected.requires_grad:
            torch.testing.assert_close(result.grad, expected.grad, atol=1e-5, rtol=1e-4)


def test_neftune_is_inactive_during_evaluation_and_validates_alpha():
    model = make_model()
    model.eval()
    ids = torch.tensor([[1, 5, 6, 7]])
    labels = torch.tensor([[-100, -100, 6, 7]])
    torch.manual_seed(123)
    baseline = mixtral_lora_backward(model, ids, torch.ones_like(ids), labels)
    torch.manual_seed(123)
    actual = mixtral_lora_backward(model, ids, torch.ones_like(ids), labels, neftune_noise_alpha=5)
    assert actual.loss == baseline.loss
    for alpha in (-1, float('nan'), 101, True):
        with pytest.raises(ValueError, match='neftune_noise_alpha'):
            mixtral_lora_backward(model, ids, torch.ones_like(ids), labels, neftune_noise_alpha=alpha)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('sliding_window', [None, 3])
def test_packed_mixtral_matches_separate_conversations(sliding_window):
    torch.manual_seed(71)
    config = MixtralConfig(vocab_size=32, hidden_size=64, intermediate_size=48,
                           num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                           num_local_experts=2, num_experts_per_tok=1, max_position_embeddings=64,
                           use_cache=False, output_router_logits=False, attention_dropout=0.0,
                           router_jitter_noise=0.0, sliding_window=sliding_window)
    config._attn_implementation = 'eager'
    model = MixtralForCausalLM(config).requires_grad_(False)
    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRA(layer.self_attn.q_proj)
    model.train()
    windowed = copy.deepcopy(model)
    first = torch.tensor([[1, 5, 6, 7]])
    second = torch.tensor([[1, 9, 4]])
    first_labels = torch.tensor([[-100, -100, 6, 7]])
    second_labels = torch.tensor([[-100, 9, 4]])
    losses = [model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                    use_cache=False).loss for ids, labels in [(first, first_labels), (second, second_labels)]]
    reference = (losses[0] + losses[1]) / 2
    reference.backward()
    ids = torch.cat([first, second], dim=1)
    labels = torch.cat([first_labels, second_labels], dim=1)
    actual = mixtral_lora_backward(windowed, ids, torch.ones_like(ids), labels, device='cuda',
                                   cu_seqlens=torch.tensor([0, 4, 7]))
    assert actual.loss == pytest.approx(reference.item(), abs=1e-5)
    assert windowed.config._attn_implementation == 'eager'
    for expected, result in zip(model.parameters(), windowed.parameters()):
        if expected.requires_grad:
            torch.testing.assert_close(result.grad, expected.grad, atol=1e-5, rtol=1e-4)


def test_mixtral_windowed_with_mapped_weights(tmp_path):
    safetensors = pytest.importorskip('safetensors.torch')
    from sytra_runner.helix_weights import MappedFrozenWeights
    torch.manual_seed(7)
    model = make_model()
    reference = copy.deepcopy(model)
    frozen = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
    checkpoint = tmp_path / 'mixtral.safetensors'
    safetensors.save_file(frozen, str(checkpoint))
    pager = MappedFrozenWeights(model, {name: (checkpoint, name) for name in frozen})
    ids = torch.tensor([[1, 5, 6, 7]])
    labels = torch.tensor([[-100, 5, 6, 7]])
    baseline = reference(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss
    baseline.backward()
    result = mixtral_lora_backward(model, ids, torch.ones_like(ids), labels, weight_pager=pager)
    assert result.loss == pytest.approx(baseline.item(), abs=1e-6)
    for actual, expected in zip(model.model.layers, reference.model.layers):
        torch.testing.assert_close(actual.self_attn.q_proj.lora_B.weight.grad,
                                   expected.self_attn.q_proj.lora_B.weight.grad, atol=1e-6, rtol=1e-4)
