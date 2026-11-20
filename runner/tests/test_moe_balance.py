from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
from torch import nn
from sytra_runner.performance.moe_balance import RouterBalance, balance_loss


def test_balance_masks_padding_and_propagates_router_gradients():
    logits = torch.tensor([[[3., 0.], [2., 0.], [0., 100.]]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0]])
    loss = balance_loss([logits], mask, num_experts=2, top_k=1)
    unpadded = balance_loss([logits[:, :2]], torch.ones(1, 2), num_experts=2, top_k=1)
    torch.testing.assert_close(loss, unpadded)
    assert loss.item() > 1
    loss.backward()
    assert logits.grad[:, :2].abs().sum() > 0
    assert logits.grad[:, 2].abs().sum() == 0


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(3, 2, bias=False)
        self.lora_A = nn.ModuleDict({'default': nn.Linear(3, 2, bias=False)})

    def forward(self, x):
        return self.base(x) + self.lora_A['default'](x)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(output_router_logits=False)
        layer = nn.Module()
        layer.block_sparse_moe = nn.Module()
        layer.block_sparse_moe.gate = Gate()
        self.layers = nn.ModuleList([layer])

    def forward(self, x):
        return self.layers[0].block_sparse_moe.gate(x)


class Trainer:
    def compute_loss(self, model, inputs, return_outputs=False, **_kwargs):
        logits = model(inputs['x'])
        loss = logits.square().mean()
        return (loss, {'logits': logits}) if return_outputs else loss


def test_balance_wraps_trainer_and_removes_hooks():
    model = Toy()
    trainer = Trainer()
    profile = {'is_moe': True, 'architecture': 'mixtral', 'num_experts': 2, 'experts_per_token': 1}
    inputs = {'x': torch.randn(1, 4, 3), 'attention_mask': torch.tensor([[1, 1, 1, 0]])}
    base = trainer.compute_loss(model, inputs)
    balance = RouterBalance(model, profile, .1)
    balance.attach(trainer)
    loss = trainer.compute_loss(model, inputs)
    assert loss.item() > base.item()
    loss.backward()
    assert model.layers[0].block_sparse_moe.gate.lora_A['default'].weight.grad is not None
    with pytest.raises(RuntimeError, match='attention mask'):
        trainer.compute_loss(model, {'x': inputs['x']})
    balance.close()
    assert not model.layers[0].block_sparse_moe.gate._forward_hooks


def test_balance_rejects_native_auxiliary_loss_and_unsupported_architecture():
    model = Toy()
    profile = {'is_moe': True, 'architecture': 'mixtral', 'num_experts': 2, 'experts_per_token': 1}
    model.config.output_router_logits = True
    with pytest.raises(ValueError, match='native router'):
        RouterBalance(model, profile, .1)
    model.config.output_router_logits = False
    with pytest.raises(ValueError, match='softmax routers'):
        RouterBalance(model, {**profile, 'architecture': 'deepseek'}, .1)


def test_balance_rejects_misaligned_logits_and_cleans_up_after_forward_error():
    with pytest.raises(ValueError, match='align'):
        balance_loss([torch.zeros(2, 2)], torch.ones(1, 3), 2, 1)
    model = Toy()
    trainer = Trainer()
    profile = {'is_moe': True, 'architecture': 'mixtral', 'num_experts': 2, 'experts_per_token': 1}
    balance = RouterBalance(model, profile, .1)
    balance.attach(trainer)
    with pytest.raises(KeyError):
        trainer.compute_loss(model, {'attention_mask': torch.ones(1, 2)})
    assert not balance.active and not balance.captured
    balance.close()
