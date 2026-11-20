"""Resumable, integrity-checked HTTP range transfers for Hub LFS files."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class _FullBodyFallback(Exception):
    """Raised when the origin ignores Range and replies 200 OK."""


class _RemoteChangedError(Exception):
    """Raised when ETag/Last-Modified changes during chunked or resumed transfer."""


def _redact_url_secrets(text: str) -> str:
    text = re.sub(r"(https?://)([^/\s:@]+):([^/\s@]+)@", r"\1***:***@", text)
    return re.sub(r"Bearer\s+[A-Za-z0-9._\-]+", "Bearer ***", text, flags=re.IGNORECASE)


def _response_validator(headers) -> str | None:
    etag = (headers.get("ETag") or "").strip()
    if etag:
        return f"etag:{etag}"
    last_mod = (headers.get("Last-Modified") or "").strip()
    if last_mod:
        return f"lm:{last_mod}"
    return None


def _write_state(state: Path, payload: dict[str, object]) -> None:
    temporary = state.with_name(state.name + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, state)


def _cleanup_staging(staging: Path, parts: list[Path], state: Path) -> None:
    staging.unlink(missing_ok=True)
    for part in parts:
        part.unlink(missing_ok=True)
    for i in range(8):
        staging.with_name(staging.name + f".part{i}").unlink(missing_ok=True)
    state.unlink(missing_ok=True)


class _SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urlsplit(req.full_url)[:2] != urlsplit(newurl)[:2]:
            redirected.remove_header("Authorization")
        return redirected


def _open_range(url: str, headers: dict[str, str]):
    return build_opener(_SafeRedirect()).open(Request(url, headers=headers), timeout=60)


def _download_full_200(
    url: str,
    headers: dict[str, str],
    staging: Path,
    destination: Path,
    size: int,
    sha256: str,
    parts: list[Path],
    state: Path,
) -> Path:
    """Single-stream fallback when a server does not honor HTTP byte ranges."""
    _cleanup_staging(staging, parts, state)
    req_headers = {k: v for k, v in headers.items() if k.lower() != "range"}
    digest = hashlib.sha256()
    written = 0
    try:
        with _open_range(url, req_headers) as response:
            status = getattr(response, "status", 200)
            if status not in (200, 206):
                raise RuntimeError(f"Unexpected HTTP status {status} during full transfer")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise RuntimeError("Compressed responses cannot be verified safely")
            cl_header = response.headers.get("Content-Length")
            if cl_header is not None:
                try:
                    if int(cl_header) != size:
                        raise RuntimeError(f"Content-Length mismatch: expected {size}, got {cl_header}")
                except ValueError:
                    pass
            with staging.open("wb") as output:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    written += len(chunk)
                    if written > size:
                        raise RuntimeError("Server returned more bytes than declared size")
                    digest.update(chunk)
                    output.write(chunk)
        if written != size or digest.hexdigest() != sha256.lower():
            raise RuntimeError("SHA-256 mismatch in HTTP fallback transfer")
        os.replace(staging, destination)
        return destination
    except Exception:
        _cleanup_staging(staging, parts, state)
        raise


def download_ranges(
    url: str,
    destination: Path,
    size: int,
    sha256: str,
    token: str | None = None,
    *,
    connections: int = 4,
) -> Path:
    """Resume separate ranges, then atomically publish only verified contents.

    The caller must supply immutable LFS metadata from a commit-pinned revision.
    Partial ranges are retained on recoverable transport errors and tied to that
    file identity plus remote ETag/Last-Modified validator when supplied.
    """
    if size <= 0 or len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256.lower()):
        raise ValueError("HTTP range transfer requires a size and valid SHA-256")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".sytra-download")
    state = staging.with_name(staging.name + ".json")
    count = min(max(1, connections), 8, size)
    identity: dict[str, object] = {"size": size, "sha256": sha256.lower(), "connections": count}
    parts = [staging.with_name(staging.name + f".part{i}") for i in range(count)]

    headers = {"Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for attempt in range(2):
        try:
            existing = json.loads(state.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
        existing_identity = (
            {k: existing.get(k) for k in ("size", "sha256", "connections")}
            if isinstance(existing, dict)
            else None
        )
        recorded_validator = existing.get("validator") if isinstance(existing, dict) else None
        if existing_identity != identity:
            _cleanup_staging(staging, parts, state)
            recorded_validator = None
            _write_state(state, dict(identity))

        meta_lock = threading.Lock()
        validator_holder: list[str | None] = [
            recorded_validator if isinstance(recorded_validator, str) and recorded_validator else None
        ]
        abort_range = threading.Event()

        def fetch(index: int) -> None:
            if abort_range.is_set():
                return
            start = size * index // count
            end = size * (index + 1) // count - 1
            length = end - start + 1
            part = parts[index]
            have = part.stat().st_size if part.exists() else 0
            if have > length:
                part.unlink()
                have = 0
            if have == length:
                return
            requested = start + have
            try:
                resp_ctx = _open_range(url, {**headers, "Range": f"bytes={requested}-{end}"})
            except HTTPError as exc:
                if exc.code in (401, 403, 404, 410, 416):
                    abort_range.set()
                raise RuntimeError(f"HTTP error {exc.code} while requesting byte range") from exc

            with resp_ctx as response:
                status = getattr(response, "status", 206)
                if status == 200:
                    abort_range.set()
                    raise _FullBodyFallback()
                expected_range = f"bytes {requested}-{end}/{size}"
                if status != 206 or response.headers.get("Content-Range", "").lower() != expected_range:
                    abort_range.set()
                    raise RuntimeError("Server did not honor the requested byte range")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    abort_range.set()
                    raise RuntimeError("Compressed range responses cannot be resumed safely")

                validator = _response_validator(response.headers)
                if validator:
                    with meta_lock:
                        if validator_holder[0] is None:
                            validator_holder[0] = validator
                            _write_state(state, {**identity, "validator": validator})
                        elif validator_holder[0] != validator:
                            abort_range.set()
                            raise _RemoteChangedError(
                                f"Remote content validator changed ({validator_holder[0]} -> {validator})"
                            )

                with part.open("ab") as output:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        if abort_range.is_set():
                            return
                        if output.tell() + len(chunk) > length:
                            abort_range.set()
                            raise RuntimeError("Server returned more bytes than requested")
                        output.write(chunk)
            if not abort_range.is_set() and part.stat().st_size != length:
                raise RuntimeError("HTTP range transfer ended prematurely")

        try:
            with ThreadPoolExecutor(max_workers=count) as pool:
                list(pool.map(fetch, range(count)))
            break
        except _FullBodyFallback:
            return _download_full_200(url, headers, staging, destination, size, sha256, parts, state)
        except _RemoteChangedError:
            _cleanup_staging(staging, parts, state)
            if attempt == 0:
                continue
            raise RuntimeError("Remote resource changed repeatedly during transfer")
        except Exception as exc:
            if abort_range.is_set():
                _cleanup_staging(staging, parts, state)
            msg = _redact_url_secrets(str(exc))
            if msg != str(exc):
                raise RuntimeError(msg) from None
            raise

    digest = hashlib.sha256()
    with staging.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
                    output.write(chunk)
    if staging.stat().st_size != size or digest.hexdigest() != sha256.lower():
        _cleanup_staging(staging, parts, state)
        raise RuntimeError("SHA-256 mismatch in HTTP range transfer")
    os.replace(staging, destination)
    _cleanup_staging(staging, parts, state)
    return destination
