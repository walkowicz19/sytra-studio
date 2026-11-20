import json
import subprocess
import sys
from pathlib import Path

import pytest

from sytra_runner.dataset_fabric import merge_datasets, normalize_row, resolve_manifest, split_dataset


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_normalizes_multiturn_and_alpaca_without_losing_content():
    row = {"conversations": [{"from": "human", "value": "first"},
                             {"from": "gpt", "value": "answer"},
                             {"from": "human", "value": "follow-up"},
                             {"from": "gpt", "value": "final"}]}
    messages = normalize_row(row)["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in messages] == ["first", "answer", "follow-up", "final"]
    alpaca = normalize_row({"instruction": "do", "input": "context", "output": "done"})
    assert alpaca["messages"][0]["content"] == "do\n\ncontext"


def test_weighted_merge_exact_quotas_and_indexed_shards(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_rows(a, [{"prompt": f"a{i}", "completion": "ok"} for i in range(10)])
    write_rows(b, [{"prompt": f"b{i}", "completion": "ok"} for i in range(10)])
    manifest = merge_datasets([a, b], tmp_path / "out", weights=[3, 1], total_samples=8, shard_rows=3)
    assert manifest["counts"] == [6, 2]
    assert [item["rows"] for item in manifest["shards"]] == [3, 3, 2]
    rows = [json.loads(line) for item in manifest["shards"]
            for line in (tmp_path / "out" / item["path"]).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8
    assert all(row["messages"][-1]["role"] == "assistant" for row in rows)
    files = resolve_manifest(tmp_path / "out" / "manifest.json")
    assert [Path(path).name for path in files] == [item["path"] for item in manifest["shards"]]


@pytest.mark.parametrize("bad_path", ["../escape.jsonl", "nested/data.jsonl", "data.parquet"])
def test_manifest_rejects_paths_outside_its_shard_set(tmp_path, bad_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "rows": 1, "shards": [{"path": bad_path, "rows": 1}]}))
    with pytest.raises(ValueError, match="Shard paths"):
        resolve_manifest(manifest)


def test_manifest_rejects_duplicate_shards_and_inconsistent_counts(tmp_path):
    (tmp_path / "one.jsonl").write_text('{}\n')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "rows": 2, "shards": [{"path": "one.jsonl", "rows": 1}] * 2}))
    with pytest.raises(ValueError, match="duplicate"):
        resolve_manifest(manifest)
    manifest.write_text(json.dumps({"version": 1, "rows": 3, "shards": [{"path": "one.jsonl", "rows": 1}]}))
    with pytest.raises(ValueError, match="total"):
        resolve_manifest(manifest)
    manifest.write_text(json.dumps({"version": 1, "rows": 2, "shards": [{"path": "one.jsonl", "rows": 2}]}))
    with pytest.raises(ValueError, match="row count mismatch"):
        resolve_manifest(manifest)


def test_failed_merge_does_not_publish_partial_output(tmp_path):
    source = tmp_path / "data.jsonl"
    write_rows(source, [{"prompt": "question", "completion": "answer"}])
    with pytest.raises(ValueError, match="Insufficient samples"):
        merge_datasets([source], tmp_path / "out", weights=[1], total_samples=2)
    assert not (tmp_path / "out").exists()


def test_curation_is_opt_in_and_reports_exact_near_and_loop_removals(tmp_path):
    source = tmp_path / "data.jsonl"
    long_answer = " ".join(f"word{i}" for i in range(100))
    near_answer = long_answer.replace("word50", "changed")
    original = {"prompt": "explain", "completion": long_answer}
    loop = {"prompt": "repeat", "completion": "one two three " * 20}
    write_rows(source, [original, original, {"prompt": "explain", "completion": near_answer}, loop])
    unfiltered = merge_datasets([source], tmp_path / "unfiltered")
    assert unfiltered["rows"] == 4 and "curation" not in unfiltered
    curated = merge_datasets([source], tmp_path / "curated", curation={
        "exact_duplicates": True, "near_duplicates": True, "repetitive": True, "similarity": 0.85,
    })
    assert curated["rows"] == 1
    assert curated["curation"]["removed"] == {
        "exact_duplicate": 1, "near_duplicate": 1, "repetitive": 1, "too_short": 0, "benchmark_overlap": 0,
    }
    assert len(source.read_text().splitlines()) == 4


def test_opt_in_invalid_conversation_filter_reports_removals(tmp_path):
    source = tmp_path / 'rows.jsonl'
    rows = [
        {'prompt': 'question', 'completion': 'answer'},
        {'prompt': 'orphan', 'completion': ''},
        {'prompt': '   ', 'completion': 'answer'},
        {'messages': [{'role': 'assistant', 'content': 'orphan answer'}]},
        {'messages': [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'world'}]},
    ]
    write_rows(source, rows)
    before = source.read_bytes()
    with pytest.raises(ValueError, match='non-empty assistant'):
        merge_datasets([source], tmp_path / 'strict')
    result = merge_datasets([source], tmp_path / 'filtered', curation={'drop_invalid': True})
    assert result['rows'] == 2
    assert result['curation']['removed']['invalid_structure'] == 3
    assert source.read_bytes() == before
    assert not (tmp_path / 'strict').exists()


def test_invalid_json_still_aborts_opt_in_filter_atomically(tmp_path):
    source = tmp_path / 'broken.jsonl'
    source.write_text('{"prompt":"hello","completion":"answer"}\n{not-json}\n')
    with pytest.raises(ValueError, match='broken.jsonl:2'):
        merge_datasets([source], tmp_path / 'out', curation={'drop_invalid': True})
    assert not (tmp_path / 'out').exists()


def test_invalid_conversation_filter_counts_parquet_rows(tmp_path):
    arrow = pytest.importorskip('pyarrow')
    parquet = pytest.importorskip('pyarrow.parquet')
    source = tmp_path / 'rows.parquet'
    parquet.write_table(arrow.table({'prompt': ['valid', 'missing'], 'completion': ['answer', None]}), source)
    result = merge_datasets([source], tmp_path / 'out', curation={'drop_invalid': True})
    assert result['rows'] == 1
    assert result['curation']['removed']['invalid_structure'] == 1


def test_weighted_curation_fails_if_survivors_cannot_meet_quota(tmp_path):
    source = tmp_path / "data.jsonl"
    write_rows(source, [{"prompt": "q", "completion": "a"}] * 3)
    with pytest.raises(ValueError, match="Insufficient samples"):
        merge_datasets([source], tmp_path / "out", weights=[1], total_samples=2,
                       curation={"exact_duplicates": True})
    assert not (tmp_path / "out").exists()


def test_curation_rejects_unknown_or_invalid_options(tmp_path):
    source = tmp_path / "data.jsonl"
    write_rows(source, [{"prompt": "q", "completion": "a"}])
    with pytest.raises(ValueError, match="Unknown curation"):
        merge_datasets([source], tmp_path / "out", curation={"unknown": True})


def test_benchmark_filter_checks_supplied_text_and_records_provenance(tmp_path):
    benchmark = tmp_path / "benchmark.jsonl"
    passage = " ".join(f"word{i}" for i in range(15))
    write_rows(benchmark, [{"question": passage, "choices": ["unrelated"]}])
    source = tmp_path / "data.jsonl"
    write_rows(source, [{"prompt": passage.upper(), "completion": "answer"},
                        {"prompt": "a different prompt", "completion": "a different answer"}])
    result = merge_datasets([source], tmp_path / "out", curation={"benchmark_files": [str(benchmark)]})
    assert result["rows"] == 1
    assert result["curation"]["removed"]["benchmark_overlap"] == 1
    info = result["curation"]["benchmarks"][0]
    assert info["path"] == str(benchmark.resolve())
    assert len(info["sha256"]) == 64 and info["new_13grams"] == 3


def test_benchmark_without_13_word_sequences_does_not_claim_coverage(tmp_path):
    source, benchmark = tmp_path / "data.jsonl", tmp_path / "benchmark.jsonl"
    write_rows(source, [{"prompt": "q", "completion": "a"}])
    write_rows(benchmark, [{"question": "too short"}])
    with pytest.raises(ValueError, match="no text with 13 words"):
        merge_datasets([source], tmp_path / "out", curation={"benchmark_files": [str(benchmark)]})
    assert not (tmp_path / "out").exists()


def test_parquet_benchmark_is_indexed(tmp_path):
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    benchmark = tmp_path / "benchmark.parquet"
    passage = " ".join(f"token{i}" for i in range(13))
    parquet.write_table(arrow.table({"question": [passage]}), benchmark)
    source = tmp_path / "data.jsonl"
    write_rows(source, [{"prompt": passage, "completion": "answer"}, {"prompt": "other", "completion": "answer"}])
    result = merge_datasets([source], tmp_path / "out", curation={"benchmark_files": [str(benchmark)]})
    assert result["curation"]["removed"]["benchmark_overlap"] == 1


def test_split_is_deterministic_and_duplicates_do_not_leak(tmp_path):
    source = tmp_path / "data.jsonl"
    examples = [{"prompt": f"question {i}", "completion": "answer"} for i in range(100)]
    write_rows(source, examples + examples)
    split_dataset(source, tmp_path / "one", validation_fraction=0.25, seed=7)
    split_dataset(source, tmp_path / "two", validation_fraction=0.25, seed=7)
    train = (tmp_path / "one" / "train.jsonl").read_bytes()
    val = (tmp_path / "one" / "validation.jsonl").read_bytes()
    assert train == (tmp_path / "two" / "train.jsonl").read_bytes()
    assert val == (tmp_path / "two" / "validation.jsonl").read_bytes()
    assert set(train.splitlines()).isdisjoint(set(val.splitlines()))


def test_existing_output_is_never_replaced(tmp_path):
    source = tmp_path / "data.jsonl"
    write_rows(source, [{"prompt": "question", "completion": "answer"}])
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep").write_text("existing")
    with pytest.raises(FileExistsError):
        merge_datasets([source], out)
    assert (out / "keep").read_text() == "existing"


def test_json_bridge_produces_consumable_training_file(tmp_path):
    source = tmp_path / "source.jsonl"
    write_rows(source, [{"prompt": "question", "completion": "answer"}])
    script = Path(__file__).resolve().parents[1] / "scripts" / "dataset_fabric.py"
    result = subprocess.run([sys.executable, str(script)], input=json.dumps({
        "operation": "merge", "inputs": [str(source)], "output_dir": str(tmp_path / "out")
    }), capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["rows"] == 1
    row = json.loads((tmp_path / "out" / "dataset.jsonl").read_text(encoding="utf-8"))
    assert row["messages"] == [{"role": "user", "content": "question"},
                               {"role": "assistant", "content": "answer"}]


def test_json_bridge_forwards_explicit_benchmark_filter(tmp_path):
    source, benchmark = tmp_path / "source.jsonl", tmp_path / "benchmark.jsonl"
    passage = " ".join(f"word{i}" for i in range(13))
    write_rows(source, [{"prompt": passage, "completion": "answer"}, {"prompt": "unique", "completion": "answer"}])
    write_rows(benchmark, [{"question": passage}])
    script = Path(__file__).resolve().parents[1] / "scripts" / "dataset_fabric.py"
    result = subprocess.run([sys.executable, str(script)], input=json.dumps({
        "operation": "merge", "inputs": [str(source)], "output_dir": str(tmp_path / "out"),
        "curation": {"benchmark_files": [str(benchmark)]},
    }), capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["curation"]["removed"]["benchmark_overlap"] == 1


def test_parquet_input_is_processed_in_batches(tmp_path):
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "input.parquet"
    parquet.write_table(arrow.table({"instruction": ["question"] * 600,
                                    "output": ["answer"] * 600}), source)
    manifest = merge_datasets([source], tmp_path / "out", shard_rows=500)
    assert manifest["rows"] == 600
    assert [shard["rows"] for shard in manifest["shards"]] == [500, 100]


def test_syntax_sanitization_is_opt_in_and_tracks_provenance(tmp_path):
    source = tmp_path / "source.jsonl"
    raw_row = {
        "system": "System instructions",
        "user": "User asks <|im_end|>\x07about ```python\nprint(1)",
        "assistant": "<|im_start|>assistant\nClean response",
    }
    write_rows(source, [raw_row])
    untouched = merge_datasets([source], tmp_path / "untouched")
    untouched_row = json.loads((tmp_path / "untouched" / "dataset.jsonl").read_text(encoding="utf-8"))
    assert untouched_row["messages"][0] == {"role": "system", "content": "System instructions"}
    assert "<|im_end|>" in untouched_row["messages"][1]["content"]
    assert "sanitization" not in untouched.get("curation", {})

    sanitized = merge_datasets([source], tmp_path / "sanitized", curation={"sanitize_syntax": True})
    sanitized_row = json.loads((tmp_path / "sanitized" / "dataset.jsonl").read_text(encoding="utf-8"))
    assert sanitized["curation"]["removed"]["syntax_sanitized"] == 1
    assert sanitized["curation"]["sanitization"]["enabled"] is True
    assert "\x07" not in sanitized_row["messages"][1]["content"]
    assert "<|im_end|>" not in sanitized_row["messages"][1]["content"]
    assert sanitized_row["messages"][1]["content"].endswith("\n```")
    assert sanitized_row["messages"][2]["content"] == "Clean response"


def test_native_tokenizer_min_tokens_counts_model_tokens(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()
    (tok_dir / "config.json").write_text('{"model_type": "gpt2"}', encoding="utf-8")
    tok = tokenizers.Tokenizer(WordLevel({"[UNK]": 0, "one": 1, "two": 2, "three": 3}, unk_token="[UNK]"))
    tok.pre_tokenizer = WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")
    fast.save_pretrained(tok_dir)

    source = tmp_path / "source.jsonl"
    write_rows(source, [
        {"prompt": "hi", "completion": "ok"},
        {"prompt": "one two three", "completion": "one two three"},
    ])
    result = merge_datasets(
        [source],
        tmp_path / "out",
        curation={"min_tokens": 5, "context": {"tokenizer_path": str(tok_dir)}},
    )
    assert result["rows"] == 1
    assert result["curation"]["removed"]["too_short"] == 1
    assert result["curation"]["token_counting"] == "native_tokenizer"
