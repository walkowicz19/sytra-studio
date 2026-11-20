import copy

import pytest

torch = pytest.importorskip('torch')
safetensors = pytest.importorskip('safetensors.torch')

from sytra_runner.helix_training import windowed_lora_backward
from sytra_runner.helix_weights import MappedFrozenWeights
from test_helix_training import TinyAdapterLayer


def fixture_model():
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([TinyAdapterLayer(4), TinyAdapterLayer(4)])
    model.head = torch.nn.Linear(4, 7).requires_grad_(False)
    return model


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_mapped_frozen_weights_preserve_windowed_gradients(tmp_path, device):
    torch.manual_seed(15)
    model = fixture_model()
    reference = copy.deepcopy(model)
    frozen = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
    checkpoint = tmp_path / 'model.safetensors'
    safetensors.save_file(frozen, str(checkpoint))
    pager = MappedFrozenWeights(model, {name: (checkpoint, name) for name in frozen})
    pointers = {name: value.data_ptr() for name, value in model.named_parameters() if name in frozen}
    inputs = torch.randn(2, 4, 4)
    labels = torch.tensor([[0, 1, 2, 3], [0, 1, -100, 4]])
    baseline = windowed_lora_backward(reference.layers, reference.head, inputs, labels)
    actual = windowed_lora_backward(model.layers, model.head, inputs, labels, device=device, weight_pager=pager)
    assert pager.mapped_parameters == len(frozen)
    assert actual.loss == pytest.approx(baseline.loss, abs=1e-6)
    torch.testing.assert_close(actual.input_gradient, baseline.input_gradient, atol=1e-6, rtol=1e-5)
    for (name, value), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
        if name in frozen:
            assert value.device.type == 'cpu' and value.data_ptr() == pointers[name]
            torch.testing.assert_close(value, frozen[name])
        else:
            torch.testing.assert_close(value.grad, expected.grad, atol=1e-6, rtol=1e-5)


def test_mapping_validation_is_atomic(tmp_path):
    model = fixture_model()
    original = model.layers[0].base.weight.data_ptr()
    checkpoint = tmp_path / 'model.safetensors'
    safetensors.save_file({'good': model.layers[0].base.weight.detach().clone(),
                           'bad': torch.zeros(1)}, str(checkpoint))
    with pytest.raises(ValueError, match='shape or dtype'):
        MappedFrozenWeights(model, {'layers.0.base.weight': (checkpoint, 'good'),
                                    'layers.1.base.weight': (checkpoint, 'bad')})
    assert model.layers[0].base.weight.data_ptr() == original
    with pytest.raises(ValueError, match='frozen'):
        MappedFrozenWeights(model, {'layers.0.lora_A.weight': (checkpoint, 'good')})


def test_mapping_is_restored_when_layer_forward_fails(tmp_path):
    model = fixture_model()
    parameter = model.layers[0].base.weight
    checkpoint = tmp_path / 'weights.safetensors'
    safetensors.save_file({'weight': parameter.detach().clone()}, str(checkpoint))
    pager = MappedFrozenWeights(model, {'layers.0.base.weight': (checkpoint, 'weight')})
    mapped_pointer = parameter.data_ptr()

    def fail_forward(layer, inputs, index):
        raise RuntimeError('forward failed')

    with pytest.raises(RuntimeError, match='forward failed'):
        windowed_lora_backward(model.layers, model.head, torch.randn(1, 3, 4),
                               torch.tensor([[0, 1, 2]]), weight_pager=pager,
                               forward_layer=fail_forward, activation_memory_bytes=0,
                               activation_spill_dir=str(tmp_path))
    assert parameter.data_ptr() == mapped_pointer
    assert parameter.device.type == 'cpu'
    assert not list(tmp_path.glob('.helix-activations-*'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_pager_restores_mapped_nf4_buffers_after_gpu_window(tmp_path):
    from sytra_runner.helix_nf4_store import open_nf4_matrix, write_nf4_matrix
    model = fixture_model()
    checkpoint = tmp_path / 'frozen.safetensors'
    safetensors.save_file({'weight': model.layers[0].base.weight.detach().clone()}, str(checkpoint))
    pager = MappedFrozenWeights(model, {'layers.0.base.weight': (checkpoint, 'weight')})
    path = tmp_path / 'nf4.safetensors'
    write_nf4_matrix(torch.randn(4, 4), path)
    model.quant = open_nf4_matrix(path)
    pager.register_nf4_buffers(model)
    pointers = (model.quant.packed_weight.data_ptr(), model.quant.weight_scales.data_ptr())
    model.cuda()
    assert model.quant.packed_weight.device.type == 'cuda'
    pager.restore(model)
    model.cpu()
    assert model.quant.packed_weight.device.type == 'cpu'
    assert (model.quant.packed_weight.data_ptr(), model.quant.weight_scales.data_ptr()) == pointers
