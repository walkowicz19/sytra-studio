import pytest
import os
from types import SimpleNamespace

torch = pytest.importorskip('torch')
from torch import nn

from sytra_runner.performance.moe_lora import apply_router_lr_ratio, resolve_moe_targets, verify_moe_targets


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(self, name, nn.Linear(4, 4, bias=False))

    def forward(self, value):
        return self.o_proj(self.q_proj(value) + self.k_proj(value) + self.v_proj(value))


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ('w1', 'w2', 'w3'):
            setattr(self, name, nn.Linear(4, 4, bias=False))

    def forward(self, value):
        return self.w2(torch.nn.functional.silu(self.w1(value)) * self.w3(value))


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.block_sparse_moe = nn.Module()
        self.block_sparse_moe.gate = nn.Linear(4, 2, bias=False)
        self.block_sparse_moe.experts = nn.ModuleList([Expert(), Expert()])
        self.block_sparse_moe.shared_expert = Expert()

    def forward(self, value):
        hidden = self.self_attn(value)
        weights = self.block_sparse_moe.gate(hidden).softmax(-1)
        return (weights[..., :1] * self.block_sparse_moe.experts[0](hidden) +
                weights[..., 1:] * self.block_sparse_moe.experts[1](hidden) +
                self.block_sparse_moe.shared_expert(hidden))


class ToyMoe(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=2)
        self.layers = nn.ModuleList([Layer(), Layer()])

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        return value


PROFILE = {'is_moe': True, 'num_experts': 2}


def test_hybrid_and_full_target_exact_supported_modules():
    model = ToyMoe()
    hybrid = resolve_moe_targets(model, PROFILE, 'moe_hybrid')
    full = resolve_moe_targets(model, PROFILE, 'moe_full')
    assert len(hybrid) == 16
    assert len(full) == 28
    assert all('.experts.' not in name for name in hybrid)
    assert sum('.experts.' in name for name in full) == 12
    assert sum('.shared_expert.' in name for name in hybrid) == 6


def test_incomplete_or_fused_layout_is_rejected():
    model = ToyMoe()
    model.layers[0].block_sparse_moe.experts[0].w3 = nn.Identity()
    with pytest.raises(ValueError, match='complete Linear experts'):
        resolve_moe_targets(model, PROFILE, 'moe_full')
    model = ToyMoe()
    model.layers[1].block_sparse_moe.gate = nn.Identity()
    with pytest.raises(ValueError, match='Linear routers'):
        resolve_moe_targets(model, PROFILE, 'moe_hybrid')


def test_router_ratio_rejects_missing_adapters():
    class Trainer:
        optimizer = None
        lr_scheduler = None
    with pytest.raises(ValueError, match='no supported trainable router adapters'):
        apply_router_lr_ratio(Trainer(), ToyMoe(), .1)


def test_real_peft_adapters_cover_the_selected_targets():
    peft = pytest.importorskip('peft')
    model = ToyMoe()
    targets = resolve_moe_targets(model, PROFILE, 'moe_full')
    model = peft.inject_adapter_in_model(peft.LoraConfig(r=2, lora_alpha=4, target_modules=targets), model)
    verify_moe_targets(model, targets)
    model(torch.randn(2, 4)).sum().backward()
    grads = [param.grad for name, param in model.named_parameters() if 'lora_B' in name]
    assert len(grads) == len(targets) and all(grad is not None for grad in grads)
    assert all(not param.requires_grad for name, param in model.named_parameters() if 'lora_' not in name)


def test_router_balance_with_real_peft_modules():
    peft = pytest.importorskip('peft')
    from sytra_runner.performance.moe_balance import RouterBalance
    model = ToyMoe()
    targets = resolve_moe_targets(model, PROFILE, 'moe_hybrid')
    model = peft.inject_adapter_in_model(peft.LoraConfig(r=2, lora_alpha=4, target_modules=targets), model)

    class Trainer:
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            output = model(inputs['x'])
            loss = output.square().mean()
            return (loss, output) if return_outputs else loss

    trainer = Trainer()
    inputs = {'x': torch.randn(2, 3, 4), 'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 0]])}
    base = trainer.compute_loss(model, inputs)
    balance = RouterBalance(model, {**PROFILE, 'architecture': 'mixtral', 'experts_per_token': 1}, .1)
    try:
        balance.attach(trainer)
        loss = trainer.compute_loss(model, inputs)
        assert loss > base
        loss.backward()
        assert all(parameter.grad is not None for name, parameter in model.named_parameters()
                   if '.gate.lora_B.' in name)
    finally:
        balance.close()


@pytest.mark.skipif(os.environ.get('SYTRA_TEST_REAL_TRAINER') != '1', reason='Optional Trainer integration test')
def test_router_ratio_integrates_with_transformers_trainer(tmp_path):
    peft = pytest.importorskip('peft')
    transformers = pytest.importorskip('transformers')
    model = ToyMoe()
    targets = resolve_moe_targets(model, PROFILE, 'moe_hybrid')
    model = peft.inject_adapter_in_model(peft.LoraConfig(r=2, lora_alpha=4, target_modules=targets), model)
    args = transformers.TrainingArguments(output_dir=str(tmp_path), learning_rate=.01, use_cpu=True, report_to='none')
    trainer = transformers.Trainer(model=model, args=args)
    assert apply_router_lr_ratio(trainer, model, .1) == 4
    trainer.create_scheduler(num_training_steps=10)
    router_ids = {id(parameter) for name, parameter in model.named_parameters() if '.gate.lora_' in name}
    for group, base_lr in zip(trainer.optimizer.param_groups, trainer.lr_scheduler.base_lrs):
        if group['params']:
            assert base_lr == pytest.approx(.001 if id(group['params'][0]) in router_ids else .01)


@pytest.mark.parametrize('use_dora', [False, True])
def test_router_optimizer_groups_preserve_decay_and_scheduler_ratio(use_dora):
    peft = pytest.importorskip('peft')
    model = ToyMoe()
    targets = resolve_moe_targets(model, PROFILE, 'moe_hybrid')
    model = peft.inject_adapter_in_model(peft.LoraConfig(r=2, lora_alpha=4, target_modules=targets, use_dora=use_dora), model)
    class Trainer:
        optimizer = None
        lr_scheduler = None
        def create_optimizer(self):
            parameters = [p for p in model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW([{'params': parameters[::2], 'weight_decay': .1},
                                                {'params': parameters[1::2], 'weight_decay': 0}], lr=.01)
            return self.optimizer
    trainer = Trainer()
    assert apply_router_lr_ratio(trainer, model, .1) == (6 if use_dora else 4)
    router_ids = {id(parameter) for name, parameter in model.named_parameters() if '.gate.lora_' in name}
    for group in trainer.optimizer.param_groups:
        if not group['params']:
            continue
        is_router = id(group['params'][0]) in router_ids
        assert all((id(parameter) in router_ids) == is_router for parameter in group['params'])
        assert group['lr'] == pytest.approx(.001 if is_router else .01)
        assert group['weight_decay'] in (0, .1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda step: .5 ** step)
    model(torch.randn(2, 4)).sum().backward()
    trainer.optimizer.step()
    scheduler.step()
    rates = {round(group['lr'], 8) for group in trainer.optimizer.param_groups if group['params']}
    assert rates == {.0005, .005}
    optimizer_state, scheduler_state = trainer.optimizer.state_dict(), scheduler.state_dict()
    resumed = Trainer()
    apply_router_lr_ratio(resumed, model, .1)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed.optimizer, lambda step: .5 ** step)
    resumed.optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)
    resumed.optimizer.step()
    resumed_scheduler.step()
    rates = {round(group['lr'], 8) for group in resumed.optimizer.param_groups if group['params']}
    assert rates == {.00025, .0025}
