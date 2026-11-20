import io
import json
import urllib.error
import urllib.parse

import pytest

from sytra_runner.web_search import search_web


class _FakeResp(io.BytesIO):
    pass


def test_search_uses_wikimedia_api_without_a_key(monkeypatch):
    monkeypatch.delenv('BRAVE_SEARCH_API_KEY', raising=False)

    def fake_opener(request, timeout):
        url = urllib.parse.urlsplit(request.full_url)
        assert url.netloc == 'en.wikipedia.org'
        assert urllib.parse.parse_qs(url.query) == {
            'action': ['query'], 'format': ['json'], 'list': ['search'],
            'srsearch': ['local llm tools'], 'srlimit': ['1'],
        }
        assert timeout == 8.0
        assert not request.has_header('X-subscription-token')
        rows = [{'pageid': 42, 'title': 'A &amp; B', 'snippet': '<span>Clean</span> text'},
                {'pageid': 43, 'title': 'Too many', 'snippet': ''}]
        return _FakeResp(json.dumps({'query': {'search': rows}}).encode())

    result = search_web('  local   llm tools ', max_results=1, opener=fake_opener)
    assert result == {'query': 'local llm tools', 'provider': 'wikipedia', 'results': [
        {'title': 'A & B', 'url': 'https://en.wikipedia.org/?curid=42', 'snippet': 'Clean text'},
    ]}


def test_search_uses_brave_with_a_key_and_validates_links(monkeypatch):
    monkeypatch.setenv('BRAVE_SEARCH_API_KEY', 'test-key')

    def fake_opener(request, timeout):
        assert request.full_url == 'https://api.search.brave.com/res/v1/web/search?q=python&count=5'
        assert request.get_header('X-subscription-token') == 'test-key'
        rows = [{'title': 'Invalid link', 'url': 'javascript:ignored'},
                {'title': 'Python', 'url': 'https://www.python.org/', 'description': '<strong>Official</strong> site'}]
        return _FakeResp(json.dumps({'web': {'results': rows}}).encode())

    result = search_web('python', opener=fake_opener)
    assert result == {'query': 'python', 'provider': 'brave', 'results': [
        {'title': 'Python', 'url': 'https://www.python.org/', 'snippet': 'Official site'},
    ]}


@pytest.mark.parametrize('status', [302, 401, 403, 429])
def test_search_reports_provider_errors_without_retry(status, monkeypatch):
    monkeypatch.setenv('BRAVE_SEARCH_API_KEY', 'test-key')
    calls = []

    def fake_opener(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, status, 'Provider error', {}, io.BytesIO(b''))

    result = search_web('test', opener=fake_opener)
    assert len(calls) == 1
    assert result['results'] == []
    assert str(status) in result['error']
    assert 'test-key' not in json.dumps(result)


@pytest.mark.parametrize(
    'body', [b'<html>blocked</html>', b'{"error":{"code":"badrequest"}}', b' ' * (512 * 1024 + 1)],
    ids=['html', 'api-error', 'oversized'],
)
def test_search_reports_invalid_provider_responses(body, monkeypatch):
    monkeypatch.delenv('BRAVE_SEARCH_API_KEY', raising=False)
    result = search_web('test', opener=lambda *_args, **_kwargs: _FakeResp(body))
    assert result['results'] == []
    assert result['error']


def test_search_rejects_empty_or_oversized_query():
    with pytest.raises(ValueError):
        search_web('   ')
    with pytest.raises(ValueError):
        search_web('x' * 513)
