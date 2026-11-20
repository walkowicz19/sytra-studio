import pytest

from sytra_runner.tool_engine import UniversalToolEngine


def test_python_evaluation_cannot_execute_host_code(tmp_path):
    engine = UniversalToolEngine(workspace_path=str(tmp_path))
    assert 'evaluate_python' not in engine.tools
    marker = tmp_path / 'executed'
    with pytest.raises(RuntimeError, match='sandbox'):
        engine._tool_evaluate_python(f"__import__('pathlib').Path({str(marker)!r}).touch()")
    assert not marker.exists()


def test_list_directory_is_confined_to_workspace(tmp_path):
    ws = tmp_path / 'ws'
    ws.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'secret.txt').write_text('x', encoding='utf-8')
    (ws / 'ok.txt').write_text('y', encoding='utf-8')
    engine = UniversalToolEngine(workspace_path=str(ws))
    assert 'ok.txt' in engine._tool_list_directory('.')
    assert 'Access denied' in engine._tool_list_directory(str(outside))
    assert 'Access denied' in engine._tool_list_directory('../outside')


def test_extract_tool_call_handles_nested_json_and_rejects_ambiguous(tmp_path):
    engine = UniversalToolEngine(workspace_path=str(tmp_path))
    nested = '```json\n{"tool_call": {"name": "list_directory", "arguments": {"path": ".", "meta": {"a": 1}}}}\n```'
    parsed = engine.extract_tool_call(nested)
    assert parsed == {'tool_call': {'name': 'list_directory', 'arguments': {'path': '.', 'meta': {'a': 1}}}}
    ambiguous = '{"tool_call": {"name": "a", "arguments": {}}} {"tool_call": {"name": "b", "arguments": {}}}'
    assert engine.extract_tool_call(ambiguous) is None
    malformed = '{"tool_call": {"name": "list_directory", "arguments": "not-an-object"}}'
    assert engine.extract_tool_call(malformed) is None
