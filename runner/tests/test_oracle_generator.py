import pytest

from sytra_runner.oracle_generator import (
    SUPPORTED_ADAPTERS,
    OracleGenerationError,
    build_oracle_case,
)


def test_oracle_generator_accepts_the_exact_compiled_adapter_ids():
    assert SUPPORTED_ADAPTERS == {
        "sytra-kimi-k2.7-code",
        "sytra-mixtral",
        "sytra-qwen3-moe",
        "sytra-qwen2-moe",
        "sytra-olmoe",
        "sytra-granite-moe",
    }


def test_build_oracle_case_preserves_teacher_forced_positions_and_probes():
    case = build_oracle_case(
        "smoke",
        [1, 2, 3],
        [4, 5, 6],
        [0.25, -1.5, 3.0, 0.0],
        [2, 1],
        absolute_tolerance=0.05,
        relative_tolerance=0.01,
    )

    assert case["teacher_forced_predictions"] == [4, 5, 6]
    assert case["final_logit_probes"][0] == {
        "token": 2,
        "expected": 3.0,
        "absolute_tolerance": 0.05,
        "relative_tolerance": 0.01,
    }


def test_build_oracle_case_rejects_incomplete_position_coverage():
    with pytest.raises(OracleGenerationError, match="one prediction per position"):
        build_oracle_case(
            "bad",
            [1, 2],
            [3],
            [0.0, 1.0],
            [1],
            absolute_tolerance=0.05,
            relative_tolerance=0.01,
        )
