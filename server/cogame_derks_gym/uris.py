"""Read/write Coworld artifact URIs: file:// and http(s)://.

Conventions mirror the Coworld runtime contract (bitworld runtime /
Cookbook "Raw Docker Shape"):

- ``file://`` URIs strip the scheme and keep the absolute path, so the
  platform's mount pattern ``file:///coworld/out/results.json`` resolves
  to ``/coworld/out/results.json``. Parent directories are created on
  write.
- Plain scheme-less strings are treated as local paths.
- ``http(s)://`` reads are GET, writes are PUT with a Content-Type
  header (signed URLs on the hosted platform). Non-2xx raises. Both are
  bounded (30s request timeout) and retried a few times with a short
  backoff before giving up — the config read and the artifact writes
  are each the episode's whole point — and every failed attempt is
  logged to stderr.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import aiohttp

_HTTP_SCHEMES = ("http://", "https://")

HTTP_ATTEMPTS = 3
HTTP_BACKOFF_SECONDS = 0.5
HTTP_TIMEOUT_SECONDS = 30.0


def local_path(uri: str) -> Path | None:
    """The local path for a file:// URI or plain path, else None."""
    if uri.startswith("file://"):
        return Path(uri[len("file://"):])
    if "://" not in uri:
        return Path(uri)
    return None


async def _http_attempt_loop(op: str, uri: str, attempt_fn, *,
                             attempts: int, backoff_seconds: float):
    """Shared bounded-retry loop for http reads and writes.

    ``attempt_fn`` performs one attempt and either returns a result or
    raises; every failure is logged to stderr with the attempt number.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")
    last_error: Exception | None = None
    for attempt in range(attempts):
        if attempt:
            await asyncio.sleep(backoff_seconds * attempt)
        try:
            return await attempt_fn()
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            last_error = exc if isinstance(exc, IOError) else \
                IOError(f"{op} {uri} failed: {exc!r}")
            print(f"{op} {uri}: attempt {attempt + 1}/{attempts} failed: "
                  f"{last_error}", file=sys.stderr)
    assert last_error is not None
    raise last_error


async def read_uri(uri: str, *,
                   attempts: int = HTTP_ATTEMPTS,
                   backoff_seconds: float = HTTP_BACKOFF_SECONDS,
                   timeout_seconds: float = HTTP_TIMEOUT_SECONDS) -> bytes:
    path = local_path(uri)
    if path is not None:
        return path.read_bytes()
    if uri.startswith(_HTTP_SCHEMES):
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)

        async def attempt() -> bytes:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(uri) as resp:
                    if not 200 <= resp.status < 300:
                        body = (await resp.text())[:200]
                        raise IOError(
                            f"GET {uri} failed with status "
                            f"{resp.status}: {body}")
                    return await resp.read()

        return await _http_attempt_loop(
            "GET", uri, attempt,
            attempts=attempts, backoff_seconds=backoff_seconds)
    raise ValueError(f"unsupported URI scheme: {uri}")


async def write_uri(uri: str, data: bytes,
                    content_type: str = "application/octet-stream", *,
                    attempts: int = HTTP_ATTEMPTS,
                    backoff_seconds: float = HTTP_BACKOFF_SECONDS,
                    timeout_seconds: float = HTTP_TIMEOUT_SECONDS) -> None:
    path = local_path(uri)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return
    if uri.startswith(_HTTP_SCHEMES):
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)

        async def attempt() -> None:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.put(
                        uri, data=data,
                        headers={"Content-Type": content_type}) as resp:
                    if not 200 <= resp.status < 300:
                        body = (await resp.text())[:200]
                        raise IOError(
                            f"PUT {uri} failed with status "
                            f"{resp.status}: {body}")

        await _http_attempt_loop(
            "PUT", uri, attempt,
            attempts=attempts, backoff_seconds=backoff_seconds)
        return
    raise ValueError(f"unsupported URI scheme: {uri}")
