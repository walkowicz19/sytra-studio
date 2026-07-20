import pytest

from sytra_runner.backends.unsloth_sft import _to_prompt_completion


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
