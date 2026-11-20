"""Count batches consumed by training_step, excluding prefetched/evaluation data."""
from __future__ import annotations

import time
from types import MethodType


class TokenMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tokens = 0
        self.completions = 0
        self.samples = 0
        self.slots = 0
        self.started = time.perf_counter()
        self.step = 0

    def add(self, inputs):
        # Preference trainers concatenate/truncate internally. Until their final
        # batch is observable, leave throughput unavailable instead of guessing.
        for prefix in ("",):
            ids = inputs.get(prefix + "input_ids")
            if ids is None:
                continue
            mask = inputs.get(prefix + "attention_mask")
            positions = inputs.get(prefix + "position_ids")
            if mask is not None and mask.ndim == 2:
                tokens = mask.detach().sum()
            elif positions is not None or mask is None:
                tokens = ids.numel()  # Padding-free batches have no attention mask.
            else:
                continue  # A 4D attention mask is not a token-count mask.
            self.tokens = self.tokens + tokens
            self.slots += ids.numel()
            self.samples = self.samples + ((positions == 0).sum().detach() if positions is not None else ids.shape[0])
            labels = inputs.get(prefix + "labels")
            if labels is not None:
                self.completions = self.completions + (labels != -100).sum().detach()

    def snapshot(self, step):
        # Synchronize device counters only at a logging boundary.
        tokens, completions, samples = int(self.tokens), int(self.completions), int(self.samples)
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        steps = step - self.step
        available = self.slots > 0
        result = dict(tokens_s=tokens / elapsed if available else None,
                      tokens_per_second=tokens / elapsed if available else None,
                      token_count_available=available,
                      completion_tokens=completions, input_tokens=tokens,
                      samples_per_second=samples / elapsed if available else None,
                      step_time_ms=elapsed * 1000 / steps if steps > 0 else None,
                      tokens_per_optimizer_step=tokens / steps if steps > 0 else None,
                      padding_ratio=1 - tokens / self.slots if self.slots else None)
        self.reset()
        self.step = step
        return result


def instrument_trainer(trainer, meter):
    original = trainer.training_step

    def training_step(self, model, inputs, *args, **kwargs):
        result = original(model, inputs, *args, **kwargs)
        meter.add(inputs)
        return result

    trainer.training_step = MethodType(training_step, trainer)
