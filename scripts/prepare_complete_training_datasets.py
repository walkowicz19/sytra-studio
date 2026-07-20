"""Normalize complete upstream datasets into Sytra conversational JSONL.

The source repositories must already exist under ``fixtures/upstream``.  The
script keeps those snapshots untouched and writes one complete, provenance-
tagged JSONL file per SFT source under ``fixtures/complete``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "fixtures" / "upstream"
OUTPUT = ROOT / "fixtures" / "complete"


def clean_text(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def row(messages: list[dict[str, str]], source: str) -> dict[str, Any] | None:
    messages = [
        {"role": clean_text(message.get("role")), "content": clean_text(message.get("content"))}
        for message in messages
        if clean_text(message.get("role")) and clean_text(message.get("content"))
    ]
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return None
    user_content = next(
        (message["content"] for message in reversed(messages[:-1]) if message["role"] == "user"),
        "",
    )
    if not user_content:
        return None
    return {
        "prompt": user_content,
        "completion": messages[-1]["content"],
        "messages": messages,
        "source_dataset": source,
    }


def write_rows(name: str, rows: Iterable[dict[str, Any] | None]) -> tuple[int, int]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / name
    written = skipped = 0
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for item in rows:
            if item is None:
                skipped += 1
                continue
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
    return written, skipped


def magicoder_rows() -> Iterable[dict[str, Any] | None]:
    source = "ise-uiuc/Magicoder-Evol-Instruct-110K"
    path = UPSTREAM / "ise-uiuc__Magicoder-Evol-Instruct-110K" / "data-evol_instruct-decontaminated.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            yield row(
                [
                    {"role": "user", "content": item.get("instruction")},
                    {"role": "assistant", "content": item.get("response")},
                ],
                source,
            )


def codealpaca_rows() -> Iterable[dict[str, Any] | None]:
    source = "HuggingFaceH4/CodeAlpaca_20K"
    path = UPSTREAM / "HuggingFaceH4__CodeAlpaca_20K" / "data" / "train-00000-of-00001.parquet"
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["prompt", "completion"], batch_size=2048):
        for item in batch.to_pylist():
            yield row(
                [
                    {"role": "user", "content": item.get("prompt")},
                    {"role": "assistant", "content": item.get("completion")},
                ],
                source,
            )


def when2call_rows() -> Iterable[dict[str, Any] | None]:
    source = "nvidia/When2Call"
    path = UPSTREAM / "nvidia__When2Call" / "train" / "when2call_train_sft.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            messages = [dict(message) for message in item.get("messages", [])]
            tools = item.get("tools") or []
            if tools:
                parsed_tools = []
                for tool in tools:
                    try:
                        parsed_tools.append(json.loads(tool) if isinstance(tool, str) else tool)
                    except json.JSONDecodeError:
                        parsed_tools.append(tool)
                tool_prompt = "Available tools:\n" + json.dumps(parsed_tools, ensure_ascii=False)
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] = clean_text(messages[0].get("content")) + "\n\n" + tool_prompt
                else:
                    messages.insert(0, {"role": "system", "content": tool_prompt})
            yield row(messages, source)


GLAIVE_TURN = re.compile(r"(?:^|\n{2,})(USER|ASSISTANT|FUNCTION RESPONSE):\s*", re.MULTILINE)


def parse_glaive_chat(system: Any, chat: Any) -> list[dict[str, str]]:
    system_text = re.sub(r"^SYSTEM:\s*", "", clean_text(system), count=1)
    messages: list[dict[str, str]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})

    chat_text = clean_text(chat)
    matches = list(GLAIVE_TURN.finditer(chat_text))
    role_map = {"USER": "user", "ASSISTANT": "assistant", "FUNCTION RESPONSE": "tool"}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(chat_text)
        content = chat_text[match.end() : end].strip()
        content = re.sub(r"\s*<\|endoftext\|>\s*$", "", content).strip()
        if content:
            messages.append({"role": role_map[match.group(1)], "content": content})
    return messages


def glaive_rows() -> Iterable[dict[str, Any] | None]:
    source = "glaiveai/glaive-function-calling-v2"
    path = UPSTREAM / "glaiveai__glaive-function-calling-v2" / "glaive-function-calling-v2.json"
    dataset = load_dataset("json", data_files=str(path), split="train")
    for item in dataset:
        yield row(parse_glaive_chat(item.get("system"), item.get("chat")), source)


def main() -> None:
    jobs = [
        ("magicoder_evol_instruct_110k.jsonl", magicoder_rows()),
        ("codealpaca_20k.jsonl", codealpaca_rows()),
        ("when2call_sft.jsonl", when2call_rows()),
        ("glaive_function_calling_v2.jsonl", glaive_rows()),
    ]
    summary: dict[str, dict[str, int]] = {}
    for name, rows in jobs:
        written, skipped = write_rows(name, rows)
        summary[name] = {"written": written, "skipped_invalid": skipped}
        print(f"{name}: {written} rows ({skipped} invalid rows skipped)")
    (OUTPUT / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
