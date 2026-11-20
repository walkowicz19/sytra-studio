import pytest

from sytra_runner.backends.unsloth_sft import _assistant_masked_row, _to_prompt_completion, _verify_assistant_mask_collator, _verify_rslora


def test_rslora_requires_confirmed_adapter_configuration():
    from types import SimpleNamespace
    _verify_rslora(SimpleNamespace(peft_config={'default': SimpleNamespace(use_rslora=True)}), True)
    with pytest.raises(RuntimeError, match='rsLoRA'):
        _verify_rslora(SimpleNamespace(peft_config={'default': SimpleNamespace(use_rslora=False)}), True)


def test_training_rejects_invalid_noise_and_rslora_configuration(tmp_path):
    from sytra_runner.config import ConfigError, RunConfig
    from test_performance import config_raw
    raw = config_raw(tmp_path)
    raw['adapter']['use_rslora'] = 'yes'
    with pytest.raises(ConfigError, match='boolean'):
        RunConfig.from_dict(raw)
    raw['adapter']['use_rslora'] = True
    raw['train']['neftune_noise_alpha'] = float('nan')
    with pytest.raises(ConfigError, match='neftune'):
        RunConfig.from_dict(raw)
    raw['train']['neftune_noise_alpha'] = 5
    raw['train_mode'] = 'dpo'
    with pytest.raises(ConfigError, match='only supported for SFT'):
        RunConfig.from_dict(raw)


class NativeTemplateTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt=False, return_dict=False):
        assert tokenize
        roles = {"system": 10, "user": 11, "assistant": 12}
        result = [1]
        for message in messages:
            result.extend([roles[message["role"]], *[ord(char) for char in message["content"]], 2])
        if add_generation_prompt:
            result.append(roles["assistant"])
        return result


def test_multiturn_masks_every_assistant_and_preserves_full_template():
    tokenizer = NativeTemplateTokenizer()
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"}, {"role": "user", "content": "R"},
                {"role": "assistant", "content": "B"}]
    result = _assistant_masked_row({"messages": messages}, tokenizer, 100)
    assert result["input_ids"] == tokenizer.apply_chat_template(messages, tokenize=True)
    supervised = [token for token, mask in zip(result["input_ids"], result["assistant_masks"]) if mask]
    assert supervised == [ord("A"), 2, ord("B"), 2]


def test_assistant_mask_rejects_truncation_and_unstable_template():
    row = {"messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}
    with pytest.raises(ValueError, match="refusing truncation"):
        _assistant_masked_row(row, NativeTemplateTokenizer(), 4)
    class Unstable(NativeTemplateTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            result = super().apply_chat_template(messages, **kwargs)
            if kwargs.get("add_generation_prompt"):
                result.append(999)
            return result
    with pytest.raises(ValueError, match="stable assistant boundaries"):
        _assistant_masked_row(row, Unstable(), 100)
    class NoGenerationPrompt(NativeTemplateTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            kwargs["add_generation_prompt"] = False
            return super().apply_chat_template(messages, **kwargs)
    with pytest.raises(ValueError, match="stable assistant boundaries"):
        _assistant_masked_row(row, NoGenerationPrompt(), 100)


def test_rejects_collator_that_fails_to_mask_prompts():
    from types import SimpleNamespace
    labels = SimpleNamespace(reshape=lambda *_: SimpleNamespace(tolist=lambda: [1, 2, 3]))
    trainer = SimpleNamespace(train_dataset=SimpleNamespace(column_names=["assistant_masks"]),
                              args=SimpleNamespace(remove_unused_columns=False),
                              data_collator=lambda _: {"labels": labels})
    with pytest.raises(RuntimeError, match="does not preserve"):
        _verify_assistant_mask_collator(trainer)


def test_multiturn_uses_real_transformers_chat_template():
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab = {"<unk>": 0, "<s>": 1, "<user>": 2, "<assistant>": 3,
             "<eot>": 4, "question": 5, "answer": 6, "other": 7, "reply": 8}
    raw = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>")
    tokenizer.chat_template = """<s> {% for m in messages %}<{{ m['role'] }}> {{ m['content'] }} <eot> {% endfor %}{% if add_generation_prompt %}<assistant> {% endif %}"""
    messages = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "other"}, {"role": "assistant", "content": "reply"}]
    result = _assistant_masked_row({"messages": messages}, tokenizer, 100)
    assert [token for token, mask in zip(result["input_ids"], result["assistant_masks"]) if mask] == [6, 4, 8, 4]


def test_to_prompt_completion_preserves_conversational_boundary():
    row = {
        "messages": [
            {"role": "system", "content": "Be accurate."},
            {"role": "user", "content": "Fix the parser."},
            {"role": "assistant", "content": "I need the schema first."},
        ]
    }

    converted = _to_prompt_completion(row)

    assert converted == {
        "prompt": [
            {"role": "system", "content": "Be accurate."},
            {"role": "user", "content": "Fix the parser."},
        ],
        "completion": [
            {"role": "assistant", "content": "I need the schema first."},
        ],
    }


def test_to_prompt_completion_supports_canonical_string_columns():
    converted = _to_prompt_completion(
        {"prompt": "What failed?", "completion": "The build was not run."}
    )

    assert converted["prompt"] == [{"role": "user", "content": "What failed?"}]
    assert converted["completion"] == [
        {"role": "assistant", "content": "The build was not run."}
    ]


def test_to_prompt_completion_supports_conversational_canonical_columns():
    row = {
        "prompt": [
            {"role": "system", "content": "Be accurate."},
            {"role": "user", "content": "What failed?"},
        ],
        "completion": [
            {"role": "assistant", "content": "The build was not run."}
        ],
    }

    assert _to_prompt_completion(row) == row


@pytest.mark.parametrize(
    "row",
    [
        {"messages": [{"role": "user", "content": "No answer"}]},
        {
            "messages": [
                {"role": "user", "content": "Empty answer"},
                {"role": "assistant", "content": "   "},
            ]
        },
    ],
)
def test_to_prompt_completion_rejects_invalid_sft_rows(row):
    with pytest.raises(ValueError):
        _to_prompt_completion(row)
