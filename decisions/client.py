"""A small OpenAI-compatible HTTP client with keep-alive.

The point of this package is latency, so the transport matters more than
usual: a fresh TLS handshake per request costs more than the forward pass
we are trying to measure. ``urllib`` opens a new connection every call, so
this keeps one pooled ``http.client`` connection per thread instead.
"""

from __future__ import annotations

import http.client
import json
import os
import queue
import ssl
import threading
import time
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlsplit

#: Hard ceiling on requests in flight from this process, shared by every
#: client and every thread. The proxy is a shared resource, so this is a
#: semaphore rather than a convention: the ways to exceed it are all
#: accidental -- a loop over many states, two engines, permutations fanning
#: out.
MAX_IN_FLIGHT = 9
_gate = threading.BoundedSemaphore(MAX_IN_FLIGHT)


def set_max_in_flight(limit: int) -> None:
    """Change the ceiling. Callers already blocked keep the old one."""
    global MAX_IN_FLIGHT, _gate
    if limit < 1:
        raise ValueError("at least one request must be allowed in flight")
    MAX_IN_FLIGHT, _gate = limit, threading.BoundedSemaphore(limit)


class APIError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status, self.body = status, body


class OpenAIClient:
    """Connection-pooled JSON POST against an OpenAI-compatible server."""

    def __init__(self, base_url: str, api_key: str | None = None, *,
                 timeout: float = 120.0, pool_size: int = 32) -> None:
        url = urlsplit(base_url.rstrip("/"))
        if url.scheme not in ("http", "https"):
            raise ValueError(f"base_url must be http(s), got {base_url!r}")
        self._host, self._port = url.hostname, url.port
        self._https = url.scheme == "https"
        self._path = url.path  # keeps a proxy's per-tenant path prefix
        self._api_key, self._timeout = api_key, timeout
        self._pool: queue.LifoQueue = queue.LifoQueue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._context = ssl.create_default_context() if self._https else None

    @classmethod
    def from_env(cls, **kwargs: Any) -> OpenAIClient:
        """Client for the proxy this machine is already configured against."""
        base = os.environ.get("DECISIONS_BASE_URL") or os.environ.get("AIB_LLM_BASE_URL")
        key = os.environ.get("DECISIONS_API_KEY") or os.environ.get("AIB_LLM_API_KEY")
        if not base:
            raise RuntimeError("Set DECISIONS_BASE_URL (or AIB_LLM_BASE_URL)")
        return cls(base, key, **kwargs)

    def _connect(self) -> http.client.HTTPConnection:
        if self._https:
            return http.client.HTTPSConnection(self._host, self._port,
                                               timeout=self._timeout, context=self._context)
        return http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)

    @contextmanager
    def _connection(self):
        """Yield a pooled connection, or a new one.

        ``conn.decisions_reused`` says which, because only a reused one can
        have been closed by the server in the meantime.
        """
        try:
            conn = self._pool.get_nowait()
            conn.decisions_reused = True
        except queue.Empty:
            conn = self._connect()
            conn.decisions_reused = False
        broken = False
        try:
            yield conn
        except Exception:
            broken = True
            raise
        finally:
            if broken:
                conn.close()
            else:
                try:
                    self._pool.put_nowait(conn)
                except queue.Full:
                    conn.close()

    def post(self, path: str, payload: dict) -> tuple[dict, float]:
        """POST ``payload`` and return ``(response, seconds)``.

        One retry on a dead pooled socket: a keep-alive connection the
        server has since closed fails on send, not on connect. Nothing else
        is retried -- a timeout or a reset on a fresh connection may mean the
        server is already working on the request, and sending it again would
        run the inference twice.
        """
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json",
                   "Content-Length": str(len(body)),
                   "Accept": "application/json",
                   "Connection": "keep-alive"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        target = self._path + path
        for attempt in (0, 1):
            started = time.perf_counter()
            conn = None
            try:
                with _gate, self._connection() as conn:
                    conn.request("POST", target, body=body, headers=headers)
                    response = conn.getresponse()
                    raw = response.read().decode("utf-8", "replace")
                    status = response.status
            except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError):
                if attempt or conn is None or not conn.decisions_reused:
                    raise
                continue
            elapsed = time.perf_counter() - started
            if status >= 400:
                raise APIError(status, raw)
            return json.loads(raw), elapsed
        raise RuntimeError("unreachable")

    def close(self) -> None:
        while True:
            try:
                self._pool.get_nowait().close()
            except queue.Empty:
                return
