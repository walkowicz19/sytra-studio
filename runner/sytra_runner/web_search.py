"""Bounded web search through the Brave and Wikimedia JSON APIs."""
from __future__ import annotations

import html
import json
import os
import re
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

MAX_BODY = 512 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _clean_text(raw: str) -> str:
    return ' '.join(html.unescape(re.sub(r'<[^>]*>', '', raw)).split())


def _results(data: dict, provider: str, limit: int) -> list[dict[str, str]]:
    group, key = ('web', 'results') if provider == 'brave' else ('query', 'search')
    rows = data.get(group) or {}
    hits = []
    for row in rows.get(key, []):
        title = _clean_text(row['title'])[:160]
        url = (row['url'] if provider == 'brave' else
               f"https://en.wikipedia.org/?curid={int(row['pageid'])}")
        parsed = urllib.parse.urlsplit(url)
        if not title or parsed.scheme not in ('http', 'https') or not parsed.hostname or len(url) > 512:
            continue
        snippet = row.get('description' if provider == 'brave' else 'snippet') or ''
        hits.append({'title': title, 'url': url, 'snippet': _clean_text(snippet)[:280]})
        if len(hits) == limit:
            break
    return hits


def search_web(query: str, *, max_results: int = 5, timeout: float = 8.0, opener=None) -> dict[str, Any]:
    """Use Brave for broad results when configured, otherwise search Wikipedia."""
    cleaned = ' '.join(query.split())
    if not cleaned or len(cleaned) > 512:
        raise ValueError('Search query must be 1 to 512 characters.')
    limit = max(1, min(int(max_results), 8))
    key = os.environ.get('BRAVE_SEARCH_API_KEY', '').strip()
    provider = 'brave' if key else 'wikipedia'
    result: dict[str, Any] = {'query': cleaned, 'provider': provider, 'results': []}
    headers = {'Accept': 'application/json', 'User-Agent': 'SytraStudio/1.3 (https://github.com/walkowicz19/sytra-studio)'}
    if key:
        endpoint = 'https://api.search.brave.com/res/v1/web/search'
        params = {'q': cleaned, 'count': limit}
        headers['X-Subscription-Token'] = key
    else:
        endpoint = 'https://en.wikipedia.org/w/api.php'
        params = {'action': 'query', 'format': 'json', 'list': 'search', 'srsearch': cleaned, 'srlimit': limit}

    request = urllib.request.Request(f'{endpoint}?{urllib.parse.urlencode(params)}', headers=headers)
    open_response = opener or urllib.request.build_opener(_NoRedirect()).open
    try:
        with open_response(request, timeout=timeout) as response:
            body = response.read(MAX_BODY + 1)
        if len(body) > MAX_BODY:
            raise ValueError('response exceeded the size limit')
        data = json.loads(body)
        if not isinstance(data, dict) or data.get('error'):
            raise ValueError('provider returned an API error')
        result['results'] = _results(data, provider, limit)
    except urllib.error.HTTPError as exc:
        result['error'] = f'{provider} search failed (HTTP {exc.code}).'
    except OSError:
        result['error'] = f'{provider} search could not connect.'
    except (ValueError, TypeError, KeyError, AttributeError):
        result['error'] = f'{provider} search returned an invalid response.'
    return result
