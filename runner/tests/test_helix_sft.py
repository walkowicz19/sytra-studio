import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
import torch.nn.functional as F

from sytra_runner.helix_sft import HelixBatch, HelixPackedBatch, HelixDatasetBatches, HelixPackedDatasetBatches, train_helix_batches
from sytra_runner.helix_training import windowed_lora_backward
from test_helix_training import TinyAdapterLayer


def test_dataset_batches_preserve_order_and_mask_padding():
    rows = [{'input_ids': [1, 2, 3, 4], 'assistant_masks': [0, 0, 1, 1]},
            {'input_ids': [1, 5, 6], 'assistant_masks': [0, 0, 1]},
            {'input_ids': [1, 7], 'assistant_masks': [0, 1]}]
    batches = HelixDatasetBatches(rows, 2, 0)
    assert len(batches) == 2
    assert batches[0].input_ids.tolist() == [[1, 2, 3, 4], [1, 5, 6, 0]]
    assert batches[0].attention_mask.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    assert batches[0].labels.tolist() == [[-100, -100, 3, 4], [-100, -100, 6, -100]]
    assert batches[1].input_ids.tolist() == [[1, 7]]
    with pytest.raises(IndexError):
        _ = batches[2]
    with pytest.raises(ValueError, match='binary assistant masks'):
        _ = HelixDatasetBatches([{'input_ids': [1, 2], 'assistant_masks': [0, 2]}], 1, 0)[0]


def test_packed_batches_preserve_boundaries_and_never_split_conversations():
    rows = [{'input_ids': [1, 2, 3], 'assistant_masks': [1, 0, 1]},
            {'input_ids': [4, 5], 'assistant_masks': [1, 1]},
            {'input_ids': [6, 7, 8, 9], 'assistant_masks': [0, 1, 0, 1]}]
    batches = HelixPackedDatasetBatches(rows, 6)
    assert len(batches) == 2
    first = batches[0]
    assert first.input_ids.tolist() == [[1, 2, 3, 4, 5]]
    assert first.labels.tolist() == [[-100, -100, 3, -100, 5]]
    assert first.attention_mask.tolist() == [[1, 1, 1, 1, 1]]
    assert first.cu_seqlens.tolist() == [0, 3, 5]
    assert first.supervised_tokens() == 2
    assert batches[1].cu_seqlens.tolist() == [0, 4]
    assert batches[1].input_ids.tolist() == [[6, 7, 8, 9]]
    with pytest.raises(ValueError, match='fit the sequence limit'):
        HelixPackedDatasetBatches(rows, 3)
    with pytest.raises(IndexError):
        _ = batches[2]


def test_packed_checkpoint_digest_includes_conversation_boundaries():
    from sytra_runner.helix_resume import _batch_digest
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = torch.tensor([[-100, 2, -100, 4, -100, 6]])
    mask = torch.ones_like(ids)
    first = HelixPackedBatch(ids, mask, labels, torch.tensor([0, 2, 6]))
    second = HelixPackedBatch(ids, mask, labels, torch.tensor([0, 4, 6]))
    assert _batch_digest([first]) != _batch_digest([second])


def test_packed_mask_builds_metadata_without_dense_token_mask(monkeypatch):
    import torch.nn.attention.flex_attention as flex
    from sytra_runner.helix_packing import packed_attention_layout
    def reject_dense(*args, **kwargs):
        raise AssertionError('A dense token mask was constructed')
    monkeypatch.setattr(flex, 'create_mask', reject_dense)
    positions, mask = packed_attention_layout(torch.tensor([0, 4096, 8192]), 'cpu')
    assert positions.shape == (1, 8192)
    # Two 4096-token docs (32 blocks each) occupy 64x32 block slots instead of quadratic 64x64.
    assert mask.kv_indices.shape == (1, 1, 64, 32)
    assert mask.q_indices.shape == (1, 1, 64, 32)
    assert mask.kv_num_blocks.shape[-1] == 64
    _, windowed = packed_attention_layout(torch.tensor([0, 4096, 8192]), 'cpu', sliding_window=256)
    assert windowed.kv_indices.shape == (1, 1, 64, 3)
    assert windowed.q_indices.shape == (1, 1, 64, 3)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('window', [None, 2])
def test_packed_attention_isolates_documents_and_resets_positions(device, window):
    from torch.nn.attention.flex_attention import flex_attention
    from sytra_runner.helix_packing import packed_attention_layout
    boundaries = torch.tensor([0, 3, 7], dtype=torch.long)
    positions, mask = packed_attention_layout(boundaries, device, sliding_window=window)
    assert positions.cpu().tolist() == [[0, 1, 2, 0, 1, 2, 3]]
    torch.manual_seed(12)
    query = torch.randn(1, 2, 7, 8, device=device, requires_grad=True)
    key = torch.randn(1, 2, 7, 8, device=device, requires_grad=True)
    value = torch.randn(1, 2, 7, 8, device=device, requires_grad=True)
    actual = flex_attention(query, key, value, block_mask=mask)
    expected = []
    for start, end in zip(boundaries[:-1].tolist(), boundaries[1:].tolist()):
        length = end - start
        indices = torch.arange(length, device=device)
        causal = indices[:, None] >= indices[None, :]
        if window is not None:
            causal &= indices[:, None] - indices[None, :] < window
        expected.append(F.scaled_dot_product_attention(query[:, :, start:end], key[:, :, start:end],
                                                       value[:, :, start:end], attn_mask=causal))
    expected = torch.cat(expected, dim=2)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    direction = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (query, key, value), direction)
    expected_grads = torch.autograd.grad(expected, (query, key, value), direction)
    for result, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('window', [None, 32, 129])
def test_packed_sparse_tiles_match_dense_attention_across_boundaries(device, window):
    from torch.nn.attention.flex_attention import flex_attention
    from sytra_runner.helix_packing import packed_attention_layout
    boundaries = torch.tensor([0, 129, 201, 401], dtype=torch.long)
    _, mask = packed_attention_layout(boundaries, device, sliding_window=window)
    torch.manual_seed(41)
    query = torch.randn(1, 1, 401, 16, device=device, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    actual = flex_attention(query, key, value, block_mask=mask)
    positions = torch.arange(401, device=device)
    doc = torch.bucketize(positions, boundaries[1:-1].to(device), right=True)
    allowed = (doc[:, None] == doc[None, :]) & (positions[:, None] >= positions[None, :])
    if window is not None:
        allowed &= positions[:, None] - positions[None, :] < window
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    direction = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (query, key, value), direction)
    expected_grads = torch.autograd.grad(expected, (query, key, value), direction)
    for result, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(result, reference, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_compiled_packed_sparse_attention_matches_dense_gradients():
    from torch.nn.attention.flex_attention import flex_attention
    from sytra_runner.helix_packing import packed_attention_layout
    boundaries = torch.tensor([0, 129, 201, 257], dtype=torch.long)
    _, mask = packed_attention_layout(boundaries, 'cuda', sliding_window=32)
    torch.manual_seed(87)
    query = torch.randn(1, 1, 257, 16, device='cuda', requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    actual = torch.compile(flex_attention)(query, key, value, block_mask=mask)
    positions = torch.arange(257, device='cuda')
    doc = torch.bucketize(positions, boundaries[1:-1].cuda(), right=True)
    allowed = ((doc[:, None] == doc[None, :]) & (positions[:, None] >= positions[None, :]) &
               (positions[:, None] - positions[None, :] < 32))
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    direction = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (query, key, value), direction)
    expected_grads = torch.autograd.grad(expected, (query, key, value), direction)
    for result, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(result, reference, atol=3e-5, rtol=3e-5)


def test_training_accumulation_matches_full_batch_update():
    torch.manual_seed(42)
    model = SimpleNamespace(layers=torch.nn.ModuleList([TinyAdapterLayer(4), TinyAdapterLayer(4)]),
                            head=torch.nn.Linear(4, 8).requires_grad_(False),
                            embeddings=torch.randn(8, 4))
    reference = copy.deepcopy(model)
    batches = [HelixBatch(torch.tensor([[0, 1, 2, 3]]), torch.ones(1, 4, dtype=torch.long),
                          torch.tensor([[-100, 1, 2, 3]])),
               HelixBatch(torch.tensor([[0, 4, 0, 0]]), torch.tensor([[1, 1, 0, 0]]),
                          torch.tensor([[-100, 4, -100, -100]]))]
    ids = torch.cat([batch.input_ids for batch in batches])
    labels = torch.cat([batch.labels for batch in batches])
    hidden = reference.embeddings[ids]
    for layer in reference.layers:
        hidden = layer(hidden)
    loss = F.cross_entropy(reference.head(hidden[:, :-1]).reshape(-1, 8), labels[:, 1:].reshape(-1), ignore_index=-100)
    ref_optim = torch.optim.SGD(reference.layers.parameters(), lr=.1)
    loss.backward()
    ref_optim.step()

    optimizer = torch.optim.SGD(model.layers.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .5 ** step)
    events = []

    def backward(model, ids, mask, labels, **kwargs):
        return windowed_lora_backward(model.layers, model.head, model.embeddings[ids], labels,
                                      device=kwargs['device'], gradient_scale=kwargs['gradient_scale'])

    train_helix_batches(model, None, batches, optimizer, max_steps=1, accumulation_steps=2,
                        scheduler=scheduler, on_step=lambda *args: events.append(args), backward=backward)
    for actual, expected in zip(model.layers.parameters(), reference.layers.parameters()):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert events[0][1] == pytest.approx(loss.item(), abs=1e-6)
    assert events[0][0] == 1 and events[0][2] == 4
    assert optimizer.param_groups[0]['lr'] == pytest.approx(.05)
    assert all(p.grad is None for p in model.layers.parameters())


def test_failed_step_clears_partial_gradients():
    parameter = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    batch = HelixBatch(torch.tensor([[0, 1]]), torch.ones(1, 2), torch.tensor([[-100, 1]]))
    def failing_backward(*args, **kwargs):
        parameter.grad = torch.ones_like(parameter)
        raise RuntimeError('failed')
    with pytest.raises(RuntimeError, match='failed'):
        train_helix_batches(None, None, [batch], optimizer, max_steps=1, backward=failing_backward)
    assert parameter.grad is None and parameter.item() == 1
    with pytest.raises(ValueError, match='prefetch_layers'):
        train_helix_batches(None, None, [batch], optimizer, max_steps=1, prefetch_layers='yes')


def test_training_cycles_without_dropping_batches():
    parameter = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.SGD([parameter], lr=.01)
    batches = [HelixBatch(torch.tensor([[0, token]]), torch.ones(1, 2), torch.tensor([[-100, token]]))
               for token in (1, 2, 3)]
    seen = []

    def backward(model, ids, mask, labels, **kwargs):
        seen.append(ids[0, 1].item())
        (parameter * kwargs['gradient_scale']).backward()
        return SimpleNamespace(loss=1., supervised_tokens=1)

    train_helix_batches(None, None, batches, optimizer, max_steps=3, accumulation_steps=2, backward=backward)
    assert seen == [1, 2, 3, 1, 2, 3]


def test_training_reads_only_the_current_accumulation_window():
    parameter = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.SGD([parameter], lr=.01)
    accessed = []

    class LargeDataset:
        def __len__(self):
            return 1_000_000

        def __getitem__(self, index):
            accessed.append(index)
            if index not in (4, 5):
                raise AssertionError('Training read outside its accumulation window')
            return HelixBatch(torch.tensor([[0, index]]), torch.ones(1, 2),
                              torch.tensor([[-100, index]]))

    def backward(model, ids, mask, labels, **kwargs):
        (parameter * kwargs['gradient_scale']).backward()
        return SimpleNamespace(loss=1., supervised_tokens=1)

    train_helix_batches(None, None, LargeDataset(), optimizer, max_steps=3,
                        start_step=2, accumulation_steps=2, backward=backward)
    assert accessed == [4, 5, 5]


def test_batch_prefetch_overlaps_next_step_and_preserves_order():
    import threading
    parameter = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.SGD([parameter], lr=.01)
    loaded_next = threading.Event()
    seen = []

    class Dataset:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            if index == 1:
                loaded_next.set()
            return HelixBatch(torch.tensor([[0, index + 1]]), torch.ones(1, 2),
                              torch.tensor([[-100, index + 1]]))

    def backward(model, ids, mask, labels, **kwargs):
        token = ids[0, 1].item()
        if token == 1:
            assert loaded_next.wait(5), 'Next batch was not loaded during backward'
        seen.append(token)
        (parameter * kwargs['gradient_scale']).backward()
        return SimpleNamespace(loss=1., supervised_tokens=1)

    train_helix_batches(None, None, Dataset(), optimizer, max_steps=3, backward=backward, prefetch_batches=True)
    assert seen == [1, 2, 3]


def test_failed_prefetched_read_clears_partial_gradients():
    parameter = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    class Dataset:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            if index == 1:
                raise RuntimeError('prefetch read failed')
            return HelixBatch(torch.tensor([[0, 1]]), torch.ones(1, 2), torch.tensor([[-100, 1]]))

    def backward(model, ids, mask, labels, **kwargs):
        parameter.grad = torch.ones_like(parameter)
        return SimpleNamespace(loss=1., supervised_tokens=1)

    # The first optimizer step completes before the next prefetched read is needed.
    with pytest.raises(RuntimeError, match='prefetch read failed'):
        train_helix_batches(None, None, Dataset(), optimizer, max_steps=2, backward=backward, prefetch_batches=True)
    assert parameter.grad is None
    assert parameter.item() == pytest.approx(.9)


def test_resume_preserves_optimizer_rng_and_dataset_position(tmp_path):
    safetensors = pytest.importorskip('safetensors.torch')
    from sytra_runner.helix_resume import save_helix_checkpoint, restore_helix_training_state
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'config.json').write_text('{}')
    safetensors.save_file({'frozen': torch.ones(1)}, str(base / 'model.safetensors'))

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([TinyAdapterLayer(4, dropout=.2)])
            self.head = torch.nn.Linear(4, 8).requires_grad_(False)
            self.register_buffer('embeddings', torch.randn(8, 4))
            self.peft_config = {'default': SimpleNamespace(base_model_name_or_path=str(base))}
        def save_pretrained(self, path, **kwargs):
            safetensors.save_file({n: p.detach() for n, p in self.named_parameters() if p.requires_grad},
                                  str(Path(path) / 'adapter_model.safetensors'))

    def create():
        torch.manual_seed(19)
        model = Toy()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .9 ** step)
        return model, optimizer, scheduler

    batches = [HelixBatch(torch.tensor([[0, token, 4]]), torch.ones(1, 3, dtype=torch.long),
                          torch.tensor([[-100, token, 4]])) for token in (1, 2, 3)]
    def backward(model, ids, mask, labels, **kwargs):
        return windowed_lora_backward(model.layers, model.head, model.embeddings[ids], labels,
                                      gradient_scale=kwargs['gradient_scale'])
    baseline, baseline_optimizer, baseline_scheduler = create()
    torch.manual_seed(77)
    train_helix_batches(baseline, None, batches, baseline_optimizer, max_steps=3, accumulation_steps=2,
                        scheduler=baseline_scheduler, backward=backward)
    interrupted, optimizer, scheduler = create()
    torch.manual_seed(77)
    checkpoint = tmp_path / 'checkpoint-1'
    def save(step, loss, count):
        save_helix_checkpoint(checkpoint, interrupted, optimizer, scheduler, batches, step=step, accumulation_steps=2)
    train_helix_batches(interrupted, None, batches, optimizer, max_steps=1, accumulation_steps=2,
                        scheduler=scheduler, on_step=save, backward=backward, prefetch_batches=True)
    resumed, resumed_optimizer, resumed_scheduler = create()
    values = safetensors.load_file(str(checkpoint / 'adapter_model.safetensors'))
    with torch.no_grad():
        for name, parameter in resumed.named_parameters():
            if name in values:
                parameter.copy_(values[name])
    step = restore_helix_training_state(checkpoint, resumed, resumed_optimizer, resumed_scheduler, batches, accumulation_steps=2)
    train_helix_batches(resumed, None, batches, resumed_optimizer, max_steps=3, start_step=step,
                        accumulation_steps=2, scheduler=resumed_scheduler, backward=backward, prefetch_batches=True)
    for actual, expected in zip(resumed.parameters(), baseline.parameters()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert resumed_scheduler.state_dict() == baseline_scheduler.state_dict()
    with pytest.raises(ValueError, match='dataset'):
        restore_helix_training_state(checkpoint, resumed, resumed_optimizer, resumed_scheduler,
                                     batches[::-1], accumulation_steps=2)
    with pytest.raises(FileExistsError):
        save_helix_checkpoint(checkpoint, resumed, resumed_optimizer, resumed_scheduler, batches,
                              step=3, accumulation_steps=2)
    (base / 'config.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='base model'):
        restore_helix_training_state(checkpoint, resumed, resumed_optimizer, resumed_scheduler,
                                     batches, accumulation_steps=2)
