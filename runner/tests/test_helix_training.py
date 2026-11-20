import copy

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from sytra_runner.helix_training import windowed_lora_backward


class TinyAdapterLayer(torch.nn.Module):
    def __init__(self, width, dropout=0.0):
        super().__init__()
        self.base = torch.nn.Linear(width, width)
        self.base.requires_grad_(False)
        self.lora_A = torch.nn.Linear(width, 2, bias=False)
        self.lora_B = torch.nn.Linear(2, width, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return torch.tanh(self.base(x) + self.lora_B(self.lora_A(self.dropout(x))))


@pytest.mark.parametrize("dropout", [0.0, 0.25])
@pytest.mark.parametrize("chunk_size", [1, 3, 100])
def test_windowed_gradients_match_full_autograd(dropout, chunk_size):
    torch.manual_seed(101)
    reference = torch.nn.ModuleList([TinyAdapterLayer(5, dropout) for _ in range(3)])
    head = torch.nn.Linear(5, 11)
    head.requires_grad_(False)
    windows = copy.deepcopy(reference)
    window_head = copy.deepcopy(head)
    original_weights = [layer.base.weight.detach().clone() for layer in windows]
    x = torch.randn(2, 6, 5)
    labels = torch.tensor([[0, 2, -100, 4, 5, 6], [0, 1, 2, -100, -100, 3]])

    reference_input = x.clone().requires_grad_(True)
    torch.manual_seed(1234)
    hidden = reference_input
    for layer in reference:
        hidden = layer(hidden)
    loss = F.cross_entropy(head(hidden[:, :-1, :]).reshape(-1, 11), labels[:, 1:].reshape(-1), ignore_index=-100)
    loss.backward()
    expected_next_random = torch.rand(5)

    torch.manual_seed(1234)
    result = windowed_lora_backward(windows, window_head, x, labels, loss_chunk_tokens=chunk_size)
    torch.testing.assert_close(torch.rand(5), expected_next_random)
    assert result.supervised_tokens == 7
    assert result.loss == pytest.approx(loss.item(), abs=1e-6)
    torch.testing.assert_close(result.input_gradient, reference_input.grad, atol=1e-6, rtol=1e-5)
    for actual, expected, original in zip(windows, reference, original_weights):
        torch.testing.assert_close(actual.lora_A.weight.grad, expected.lora_A.weight.grad, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual.lora_B.weight.grad, expected.lora_B.weight.grad, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual.base.weight, original)
        assert actual.base.weight.grad is None


def test_windowed_rejects_unmasked_and_non_adapter_training():
    layer = TinyAdapterLayer(4)
    head = torch.nn.Linear(4, 5).requires_grad_(False)
    x = torch.randn(1, 3, 4)
    with pytest.raises(ValueError, match="No supervised tokens"):
        windowed_lora_backward([layer], head, x, torch.full((1, 3), -100, dtype=torch.long))
    layer.base.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="not a LoRA"):
        windowed_lora_backward([layer], head, x, torch.tensor([[0, 1, 2]]))


def test_windowed_gradients_accumulate_across_steps():
    torch.manual_seed(99)
    layer = TinyAdapterLayer(4)
    head = torch.nn.Linear(4, 6).requires_grad_(False)
    x = torch.randn(1, 4, 4)
    labels = torch.tensor([[0, 1, 2, 3]])
    windowed_lora_backward([layer], head, x, labels, loss_chunk_tokens=2)
    first = layer.lora_A.weight.grad.clone()
    windowed_lora_backward([layer], head, x, labels, loss_chunk_tokens=2)
    torch.testing.assert_close(layer.lora_A.weight.grad, first * 2)


def test_token_weighted_accumulation_matches_combined_batch():
    torch.manual_seed(91)
    layers = torch.nn.ModuleList([TinyAdapterLayer(4), TinyAdapterLayer(4)])
    reference = copy.deepcopy(layers)
    head = torch.nn.Linear(4, 7).requires_grad_(False)
    x = torch.randn(2, 4, 4)
    labels = torch.tensor([[0, 1, 2, 3], [0, 1, -100, -100]])
    hidden = x
    for layer in reference:
        hidden = layer(hidden)
    expected = F.cross_entropy(head(hidden[:, :-1]).reshape(-1, 7), labels[:, 1:].reshape(-1), ignore_index=-100)
    expected.backward()
    first = windowed_lora_backward(layers, head, x[:1], labels[:1], gradient_scale=3 / 4)
    second = windowed_lora_backward(layers, head, x[1:], labels[1:], gradient_scale=1 / 4)
    assert (3 * first.loss + second.loss) / 4 == pytest.approx(expected.item(), abs=1e-6)
    for actual, baseline in zip(layers, reference):
        torch.testing.assert_close(actual.lora_A.weight.grad, baseline.lora_A.weight.grad, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual.lora_B.weight.grad, baseline.lora_B.weight.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf'), True])
def test_invalid_gradient_scale_is_rejected(scale):
    layer = TinyAdapterLayer(4)
    head = torch.nn.Linear(4, 6).requires_grad_(False)
    with pytest.raises(ValueError, match='gradient_scale'):
        windowed_lora_backward([layer], head, torch.randn(1, 3, 4), torch.tensor([[0, 1, 2]]), gradient_scale=scale)


def test_optimizer_updates_only_adapters_after_windowed_backward():
    layer = TinyAdapterLayer(4)
    head = torch.nn.Linear(4, 6).requires_grad_(False)
    before_base = layer.base.weight.detach().clone()
    before_adapter = layer.lora_B.weight.detach().clone()
    optimizer = torch.optim.Adam([p for p in layer.parameters() if p.requires_grad], lr=0.01)
    windowed_lora_backward([layer], head, torch.randn(1, 4, 4), torch.tensor([[0, 1, 2, 3]]), loss_chunk_tokens=1)
    optimizer.step()
    torch.testing.assert_close(layer.base.weight, before_base)
    assert not torch.equal(layer.lora_B.weight, before_adapter)


def test_disk_spill_preserves_windowed_loss_and_gradients(tmp_path):
    torch.manual_seed(4)
    layers = torch.nn.ModuleList([TinyAdapterLayer(4), TinyAdapterLayer(4)])
    reference = copy.deepcopy(layers)
    head = torch.nn.Linear(4, 6).requires_grad_(False)
    inputs = torch.randn(2, 4, 4)
    labels = torch.tensor([[0, 1, 2, 3], [0, 4, -100, 5]])
    expected = windowed_lora_backward(reference, head, inputs, labels)
    result = windowed_lora_backward(layers, head, inputs, labels, activation_memory_bytes=0,
                                    activation_spill_dir=str(tmp_path))
    assert result.loss == pytest.approx(expected.loss, abs=1e-6)
    torch.testing.assert_close(result.input_gradient, expected.input_gradient)
    for actual, baseline in zip(layers.parameters(), reference.parameters()):
        if actual.requires_grad:
            torch.testing.assert_close(actual.grad, baseline.grad)
    assert not list(tmp_path.iterdir())


def test_prefetch_rejects_cpu_and_invalid_setting():
    layer = TinyAdapterLayer(4)
    head = torch.nn.Linear(4, 6).requires_grad_(False)
    inputs = torch.randn(1, 3, 4)
    labels = torch.tensor([[0, 1, 2]])
    with pytest.raises(ValueError, match='requires an available CUDA'):
        windowed_lora_backward([layer], head, inputs, labels, prefetch_layers=True)
    with pytest.raises(ValueError, match='must be a boolean'):
        windowed_lora_backward([layer], head, inputs, labels, prefetch_layers='yes')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('memory_bytes', [0, 1024 * 1024])
def test_cuda_prefetch_preserves_gradients_and_rng(tmp_path, memory_bytes):
    torch.manual_seed(48)
    layers = torch.nn.ModuleList([TinyAdapterLayer(4, dropout=.25) for _ in range(3)])
    reference = copy.deepcopy(layers)
    head = torch.nn.Linear(4, 7).requires_grad_(False)
    inputs = torch.randn(2, 5, 4)
    labels = torch.tensor([[0, 1, 2, 3, 4], [0, 4, -100, 5, 6]])
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(456)
    expected = windowed_lora_backward(reference, head, inputs, labels, device='cuda',
                                       activation_memory_bytes=memory_bytes, activation_spill_dir=str(tmp_path))
    expected_cpu_rng = torch.random.get_rng_state()
    expected_cuda_rng = torch.cuda.get_rng_state()
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(456)
    actual = windowed_lora_backward(layers, head, inputs, labels, device='cuda', prefetch_layers=True,
                                    activation_memory_bytes=memory_bytes, activation_spill_dir=str(tmp_path))
    assert actual.loss == pytest.approx(expected.loss, abs=1e-6)
    torch.testing.assert_close(actual.input_gradient, expected.input_gradient, atol=1e-6, rtol=1e-5)
    for parameter, baseline in zip(layers.parameters(), reference.parameters()):
        assert parameter.device.type == 'cpu'
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, baseline.grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(torch.random.get_rng_state(), expected_cpu_rng)
    torch.testing.assert_close(torch.cuda.get_rng_state(), expected_cuda_rng)
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_prefetch_releases_pending_layer_after_failure():
    layers = torch.nn.ModuleList([TinyAdapterLayer(4), TinyAdapterLayer(4)])
    head = torch.nn.Linear(4, 6).requires_grad_(False)
    class Pager:
        failed = False
        def prepare(self, layer, device):
            pass
        def restore(self, layer):
            if layer is layers[0] and not self.failed:
                self.failed = True
                raise RuntimeError('injected restore failure')
    with pytest.raises(RuntimeError, match='injected restore failure'):
        windowed_lora_backward(layers, head, torch.randn(1, 3, 4), torch.tensor([[0, 1, 2]]),
                               device='cuda', weight_pager=Pager(), prefetch_layers=True)
    assert all(parameter.device.type == 'cpu' for parameter in layers.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_prefetch_enqueues_weights_on_a_separate_stream(monkeypatch):
    import sytra_runner.helix_training as training
    layers = torch.nn.ModuleList([TinyAdapterLayer(4), TinyAdapterLayer(4), TinyAdapterLayer(4)])
    head = torch.nn.Linear(4, 6).requires_grad_(False)
    compute_stream = torch.cuda.current_stream().cuda_stream
    original = training._move_module
    transfer_streams = []
    def observe(module, device, *, non_blocking=False):
        if non_blocking and torch.device(device).type == 'cuda':
            transfer_streams.append(torch.cuda.current_stream().cuda_stream)
        return original(module, device, non_blocking=non_blocking)
    monkeypatch.setattr(training, '_move_module', observe)
    windowed_lora_backward(layers, head, torch.randn(1, 4, 4), torch.tensor([[0, 1, 2, 3]]),
                           device='cuda', prefetch_layers=True)
    assert len(transfer_streams) == 2 * (len(layers) - 1)
    assert len(set(transfer_streams)) == 1 and transfer_streams[0] != compute_stream


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_prefetch_pins_within_budget_and_skips_deferred_experts():
    from sytra_runner.helix_prefetch import CudaLayerPrefetcher
    from sytra_runner.helix_training import _move_module
    class Deferred(torch.nn.Module):
        _helix_deferred_experts = True
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, 4), requires_grad=False)
        def _apply(self, fn, recurse=True):
            return self
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = torch.nn.Linear(4, 4)
            self.experts = Deferred()
    layer = Layer()
    with CudaLayerPrefetcher('cuda', None, _move_module, pinned_bytes=32) as prefetcher:
        prefetcher.schedule(layer)
        _, event, retained, staged_bytes = prefetcher.pending
        assert 0 < staged_bytes <= 32 and prefetcher.peak_pinned_bytes <= 32
        assert retained and all(tensor.is_pinned() for tensor in retained)
        assert layer.dense.weight.device.type == 'cuda'
        assert layer.experts.weight.device.type == 'cpu' and not layer.experts.weight.is_pinned()
        prefetcher.take(layer)
        assert prefetcher.pending is None
        torch.cuda.current_stream().synchronize()
        _move_module(layer, 'cpu')
    assert all(parameter.device.type == 'cpu' for parameter in layer.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime unavailable")
def test_cuda_windowed_gradients_match_gpu_autograd():
    torch.manual_seed(12)
    reference = torch.nn.ModuleList([TinyAdapterLayer(4), TinyAdapterLayer(4)]).cuda()
    head = torch.nn.Linear(4, 7).requires_grad_(False).cuda()
    windows = copy.deepcopy(reference).cpu()
    window_head = copy.deepcopy(head).cpu()
    x = torch.randn(2, 4, 4)
    labels = torch.tensor([[0, 1, 2, 3], [0, 4, -100, 5]])
    ref_x = x.cuda().requires_grad_(True)
    hidden = ref_x
    for layer in reference:
        hidden = layer(hidden)
    loss = F.cross_entropy(head(hidden[:, :-1]).reshape(-1, 7), labels[:, 1:].cuda().reshape(-1), ignore_index=-100)
    loss.backward()
    result = windowed_lora_backward(windows, window_head, x, labels, device="cuda", loss_chunk_tokens=2)
    torch.testing.assert_close(result.input_gradient, ref_x.grad.cpu(), atol=1e-5, rtol=1e-4)
    for actual, expected in zip(windows, reference):
        torch.testing.assert_close(actual.lora_A.weight.grad, expected.lora_A.weight.grad.cpu(), atol=1e-5, rtol=1e-4)
