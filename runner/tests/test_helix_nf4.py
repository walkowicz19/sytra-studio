import copy

import pytest

torch = pytest.importorskip('torch')
import torch.nn.functional as F

from sytra_runner.helix_nf4 import NF4_VALUES, FrozenNF4Linear, NF4LoRALinear, quantize_nf4


def reference_matrix(packed, scales, rows, cols, block_size):
    encoded = packed.cpu().tolist()
    factors = scales.cpu().tolist()
    values = [NF4_VALUES[(encoded[i // 2] >> (4 * (i % 2))) & 15] * factors[i // block_size]
              for i in range(rows * cols)]
    return torch.tensor(values, dtype=torch.float32).view(rows, cols)


def test_nf4_chunked_quantization_and_odd_block_boundaries():
    torch.manual_seed(41)
    weights = torch.randn(7, 13)
    packed, scales = quantize_nf4(weights, block_size=8, blocks_per_chunk=1)
    packed2, scales2 = quantize_nf4(weights, block_size=8, blocks_per_chunk=100)
    torch.testing.assert_close(packed, packed2, rtol=0, atol=0)
    torch.testing.assert_close(scales, scales2, rtol=0, atol=0)
    assert packed.numel() == (weights.numel() + 1) // 2
    assert scales.numel() == (weights.numel() + 7) // 8
    reconstructed = reference_matrix(packed, scales, 7, 13, 8)
    assert (weights - reconstructed).square().mean() < weights.square().mean() * .02
    zero = FrozenNF4Linear.from_float(torch.zeros(3, 5), block_size=8)
    torch.testing.assert_close(zero(torch.randn(2, 5)), torch.zeros(2, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_triton_decoder_matches_torch_for_unaligned_ranges(monkeypatch, dtype):
    pytest.importorskip('triton')
    from sytra_runner.helix_nf4 import _decode_range
    torch.manual_seed(4)
    packed, scales = quantize_nf4(torch.randn(73, 31), block_size=8)
    packed, scales = packed.cuda(), scales.cuda()
    codebook = torch.tensor(NF4_VALUES, dtype=torch.float32, device='cuda')
    for start, end in [(0, 1), (1, 17), (7, 23), (31, 2100), (2101, 2263)]:
        monkeypatch.delenv('SYTRA_HELIX_TRITON_NF4', raising=False)
        expected = _decode_range(packed, scales, start, end, 8, codebook, dtype)
        monkeypatch.setenv('SYTRA_HELIX_TRITON_NF4', '1')
        actual = _decode_range(packed, scales, start, end, 8, codebook, dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('rows,cols,tokens', [(73, 31, 3), (67, 129, 23), (33, 65, 37)])
def test_triton_nf4_matmul_and_backward_match_reference(monkeypatch, dtype, rows, cols, tokens):
    pytest.importorskip('triton')
    monkeypatch.setenv('SYTRA_HELIX_TRITON_NF4_MATMUL', '1')
    torch.manual_seed(19)
    layer = FrozenNF4Linear.from_float(torch.randn(rows, cols), block_size=8).cuda()
    weight = reference_matrix(layer.packed_weight, layer.weight_scales, rows, cols, 8).to('cuda', dtype=dtype)
    # Exercise an input view and a sliced, non-contiguous upstream gradient.
    inputs = torch.randn(tokens, cols * 2, device='cuda', dtype=dtype)[:, ::2].detach().requires_grad_(True)
    ref_input = inputs.detach().clone().requires_grad_(True)
    direction = torch.randn(tokens, rows * 2, device='cuda', dtype=dtype)[:, ::2]
    output = layer(inputs)
    expected = F.linear(ref_input, weight)
    output.backward(direction)
    expected.backward(direction)
    tolerance = 3e-2 if dtype == torch.bfloat16 else 3e-3 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(output, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(inputs.grad, ref_input.grad, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_nf4_linear_forward_and_input_gradient_match_independent_reference(device, dtype):
    torch.manual_seed(9)
    weights = torch.randn(11, 13)
    layer = FrozenNF4Linear.from_float(weights, block_size=8, chunk_rows=3).to(device)
    reference = reference_matrix(layer.packed_weight, layer.weight_scales, 11, 13, 8).to(device=device, dtype=dtype)
    actual_input = torch.randn(2, 3, 13, device=device, dtype=dtype, requires_grad=True)
    ref_input = actual_input.detach().clone().requires_grad_(True)
    output = layer(actual_input)
    expected = F.linear(ref_input, reference)
    tolerance = 2e-3 if dtype == torch.float16 else 1e-6
    torch.testing.assert_close(output, expected, atol=tolerance, rtol=tolerance)
    direction = torch.randn_like(output)
    (output * direction).sum().backward()
    (expected * direction).sum().backward()
    torch.testing.assert_close(actual_input.grad, ref_input.grad, atol=tolerance, rtol=tolerance)
    assert not list(layer.parameters())


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('rslora', [False, True])
def test_nf4_lora_gradients_match_dequantized_base(device, rslora):
    torch.manual_seed(12)
    base = FrozenNF4Linear.from_float(torch.randn(9, 7), block_size=8, chunk_rows=2).to(device)
    layer = NF4LoRALinear(base, rank=3, alpha=6, use_rslora=rslora).to(device)
    torch.nn.init.normal_(layer.lora_B.weight)
    a = copy.deepcopy(layer.lora_A)
    b = copy.deepcopy(layer.lora_B)
    quant_weight = reference_matrix(base.packed_weight, base.weight_scales, 9, 7, 8).to(device)
    inputs = torch.randn(4, 7, device=device, requires_grad=True)
    reference_input = inputs.detach().clone().requires_grad_(True)
    result = layer(inputs)
    expected = F.linear(reference_input, quant_weight) + b(a(reference_input)) * layer.scaling
    direction = torch.randn_like(result)
    (result * direction).sum().backward()
    (expected * direction).sum().backward()
    torch.testing.assert_close(inputs.grad, reference_input.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(layer.lora_A.weight.grad, a.weight.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(layer.lora_B.weight.grad, b.weight.grad, atol=1e-5, rtol=1e-5)


def test_nf4_backward_saves_compressed_weights_only():
    layer = FrozenNF4Linear.from_float(torch.randn(64, 65), block_size=64, chunk_rows=4)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value, lambda value: value):
        layer(torch.randn(2, 65, requires_grad=True)).sum().backward()
    assert {item.dtype for item in saved} == {torch.uint8, torch.float32}
    assert sum(item.numel() * item.element_size() for item in saved) < 64 * 65 * 4


def test_nf4_rejects_invalid_and_nonfinite_inputs():
    with pytest.raises(ValueError, match='NaN or infinite'):
        quantize_nf4(torch.tensor([[float('nan')]]))
    with pytest.raises(ValueError, match='block size'):
        quantize_nf4(torch.ones(2, 3), block_size=3)
    with pytest.raises(ValueError, match='scales'):
        FrozenNF4Linear(torch.zeros(2, dtype=torch.uint8), torch.tensor([float('nan')]), 2, 2)
    layer = FrozenNF4Linear.from_float(torch.ones(2, 2)).half()
    with pytest.raises(ValueError, match='scales must remain float32'):
        layer(torch.ones(1, 2, dtype=torch.float16))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_nf4_lora_windowed_backward_matches_full_autograd(device):
    from sytra_runner.helix_training import windowed_lora_backward
    torch.manual_seed(99)
    layers = torch.nn.ModuleList([
        torch.nn.Sequential(NF4LoRALinear(FrozenNF4Linear.from_float(torch.randn(7, 7), block_size=8,
                                                                    chunk_rows=2), rank=2, alpha=4), torch.nn.Tanh())
        for _ in range(2)])
    for layer in layers:
        torch.nn.init.normal_(layer[0].lora_B.weight)
    head = torch.nn.Linear(7, 11).requires_grad_(False)
    reference = copy.deepcopy(layers).to(device)
    ref_head = copy.deepcopy(head).to(device)
    inputs = torch.randn(2, 4, 7)
    labels = torch.tensor([[0, 1, 2, 3], [0, 2, -100, 4]])
    ref_input = inputs.to(device).requires_grad_(True)
    hidden = ref_input
    for layer in reference:
        hidden = layer(hidden)
    loss = F.cross_entropy(ref_head(hidden[:, :-1]).reshape(-1, 11), labels[:, 1:].to(device).reshape(-1), ignore_index=-100)
    loss.backward()
    result = windowed_lora_backward(layers, head, inputs, labels, device=device)
    assert result.loss == pytest.approx(loss.item(), abs=1e-6)
    torch.testing.assert_close(result.input_gradient, ref_input.grad.cpu(), atol=1e-6, rtol=1e-5)
    for actual, expected in zip(layers.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad.cpu(), atol=1e-6, rtol=1e-5)


def test_streamed_nf4_file_matches_in_memory_quantization(tmp_path):
    from sytra_runner.helix_nf4_store import open_nf4_matrix, write_nf4_matrix
    torch.manual_seed(48)
    weights = torch.randn(7, 13)
    original = weights.clone()
    path = tmp_path / 'expert.safetensors'
    assert write_nf4_matrix(weights, path, block_size=8, blocks_per_chunk=1) == path
    mapped = open_nf4_matrix(path, chunk_rows=2)
    direct = FrozenNF4Linear.from_float(weights, block_size=8, chunk_rows=2)
    torch.testing.assert_close(mapped.packed_weight, direct.packed_weight, atol=0, rtol=0)
    torch.testing.assert_close(mapped.weight_scales, direct.weight_scales, atol=0, rtol=0)
    inputs = torch.randn(3, 13)
    torch.testing.assert_close(mapped(inputs), direct(inputs), atol=0, rtol=0)
    torch.testing.assert_close(weights, original, atol=0, rtol=0)
    original_bytes = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_nf4_matrix(weights, path, block_size=8)
    assert path.read_bytes() == original_bytes
    assert not list(tmp_path.glob('.sytra-nf4-*'))


def test_streamed_nf4_failure_does_not_leave_partial_file(tmp_path):
    from sytra_runner.helix_nf4_store import write_nf4_matrix
    weights = torch.ones(3, 7)
    weights[-1, -1] = float('nan')
    with pytest.raises(ValueError, match='NaN or infinite'):
        write_nf4_matrix(weights, tmp_path / 'invalid.safetensors', block_size=8, blocks_per_chunk=1)
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_mapped_nf4_matrix_can_train_on_cuda(tmp_path):
    from sytra_runner.helix_nf4_store import open_nf4_matrix, write_nf4_matrix
    torch.manual_seed(5)
    path = tmp_path / 'weights.safetensors'
    write_nf4_matrix(torch.randn(9, 13), path, block_size=8, blocks_per_chunk=2)
    layer = open_nf4_matrix(path, chunk_rows=2).cuda()
    weights = reference_matrix(layer.packed_weight, layer.weight_scales, 9, 13, 8).cuda()
    inputs = torch.randn(4, 13, device='cuda', requires_grad=True)
    reference_input = inputs.detach().clone().requires_grad_(True)
    output = layer(inputs)
    expected = F.linear(reference_input, weights)
    output.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(inputs.grad, reference_input.grad, atol=1e-5, rtol=1e-5)


def test_nf4_peft_custom_dispatch_and_adapter_gradients(monkeypatch):
    peft = pytest.importorskip('peft')
    import peft.tuners.lora.model as lora_model
    from sytra_runner.helix_nf4_peft import NF4PeftLinear
    monkeypatch.setattr(lora_model, 'is_bnb_available', lambda: False)
    monkeypatch.setattr(lora_model, 'is_bnb_4bit_available', lambda: False)
    torch.manual_seed(33)
    base = FrozenNF4Linear.from_float(torch.randn(9, 7), block_size=8)
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = base
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        def forward(self, inputs):
            return self.layer(inputs)
    model = Toy()
    config = peft.LoraConfig(r=2, lora_alpha=4, target_modules=['layer'])
    config._register_custom_module({FrozenNF4Linear: NF4PeftLinear})
    peft.inject_adapter_in_model(config, model)
    assert isinstance(model.layer, NF4PeftLinear)
    result = model(torch.randn(3, 7, requires_grad=True))
    result.square().mean().backward()
    assert model.layer.lora_B['default'].weight.grad is not None
    assert not model.layer.base_layer.packed_weight.requires_grad
    with pytest.raises(RuntimeError, match='not supported'):
        model.layer.merge()


def test_nf4_cache_reuses_source_and_rejects_corruption(tmp_path):
    import gc
    from sytra_runner.helix_nf4_store import cached_nf4_matrix
    weights = torch.arange(35, dtype=torch.float32).view(5, 7)
    first = cached_nf4_matrix(weights, tmp_path, block_size=8)
    files = list(tmp_path.glob('*.safetensors'))
    assert len(files) == 1 and len(list(tmp_path.glob('*.sha256'))) == 1
    initial = files[0].stat().st_mtime_ns
    second = cached_nf4_matrix(weights, tmp_path, block_size=8)
    torch.testing.assert_close(first.packed_weight, second.packed_weight)
    assert files[0].stat().st_mtime_ns == initial
    del first, second
    gc.collect()
    with files[0].open('r+b') as handle:
        handle.seek(-1, 2)
        value = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match='checksum mismatch'):
        cached_nf4_matrix(weights, tmp_path, block_size=8)


def test_nf4_cache_rejects_source_changed_during_conversion(tmp_path, monkeypatch):
    from sytra_runner import helix_nf4_store as store
    weights = torch.ones(4, 4)
    original = store.write_nf4_matrix
    def change_source(weight, path, **kwargs):
        result = original(weight, path, **kwargs)
        weight[0, 0] = 2
        return result
    monkeypatch.setattr(store, 'write_nf4_matrix', change_source)
    with pytest.raises(RuntimeError, match='Source weights changed'):
        store.cached_nf4_matrix(weights, tmp_path, block_size=8)
    assert not list(tmp_path.iterdir())
