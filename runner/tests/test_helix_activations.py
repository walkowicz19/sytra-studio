import pytest

torch = pytest.importorskip('torch')
from sytra_runner.helix_activations import ActivationStash


def test_stash_spills_and_removes_temporary_files(tmp_path):
    first = torch.arange(4, dtype=torch.float32)
    second = torch.arange(6, dtype=torch.float32)
    with ActivationStash(16, directory=tmp_path) as stash:
        stash.push(first)
        stash.push(second)
        assert stash.resident_bytes == 16
        assert list(tmp_path.iterdir())
        torch.testing.assert_close(stash.last(), second)
        torch.testing.assert_close(stash.pop(), second)
        torch.testing.assert_close(stash.pop(), first)
        assert stash.resident_bytes == 0
    assert not list(tmp_path.iterdir())


def test_stash_removes_spills_after_failure(tmp_path):
    with pytest.raises(RuntimeError, match='failure'):
        with ActivationStash(0, directory=tmp_path) as stash:
            stash.push(torch.ones(3))
            raise RuntimeError('failure')
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_stash_copies_to_pinned_memory_and_waits_for_event(tmp_path):
    value = torch.arange(100, dtype=torch.float32, device='cuda')
    with ActivationStash(1024, device='cuda', directory=tmp_path) as stash:
        stash.push(value)
        entry = stash._entries[-1]
        assert entry.tensor is not None and entry.tensor.is_pinned() and entry.event is not None
        torch.testing.assert_close(stash.pop(), value.cpu())
    assert not list(tmp_path.iterdir())
