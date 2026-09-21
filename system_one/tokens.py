"""Finding single-token option labels without a local tokenizer.

Upstream loads the HF tokenizer and asks it which indexes encode to a
single token. We deliberately cannot: the model lives inside a
confidential-computing enclave and all we have is an OpenAI-compatible
API. So we use the server's own tokenizer as an oracle.

vLLM's ``/completions`` accepts ``echo`` (return the prompt) together
with ``logprobs`` (return per-token detail) and the vLLM extra
``return_tokens_as_token_ids``. Echoing a zero-token completion is a
prefill-only round trip that hands back the exact tokenization of the
prompt -- ids with the flag, text without it. That is a complete
``encode()``, and it is authoritative for the model actually serving,
which a locally downloaded tokenizer is only ever assumed to be.

Results are cached on disk, keyed by model, so this costs two requests
once per model rather than once per process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .client import OpenAIClient

CACHE_VERSION = 3


def cache_dir() -> Path:
    root = os.environ.get("SYSTEM_ONE_CACHE") or (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "system_one")
    return Path(root)


class TokenOracle:
    """Tokenizes through the serving endpoint and caches what it learns."""

    def __init__(self, client: OpenAIClient, model: str, *, cache: bool = True) -> None:
        self.client, self.model = client, model
        self._cache_path = cache_dir() / f"{model.replace('/', '_')}.json"
        self._memo: dict[str, list[int]] = {}
        self._indexes: dict[str, list[int]] = {}
        if cache and self._cache_path.exists():
            try:
                stored = json.loads(self._cache_path.read_text())
                if stored.get("version") == CACHE_VERSION:
                    self._memo = {k: v for k, v in stored.get("encode", {}).items()}
                    self._indexes = stored.get("indexes", {})
            except (OSError, ValueError):
                pass
        self._cache = cache

    def _save(self) -> None:
        if not self._cache:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(
                {"version": CACHE_VERSION, "encode": self._memo, "indexes": self._indexes}))
        except OSError:
            pass

    def _echo(self, prompts: list[str], *, ids: bool) -> list[list]:
        payload = {"model": self.model, "prompt": prompts, "max_tokens": 0,
                   "echo": True, "logprobs": 0, "temperature": 0}
        if ids:
            payload["return_tokens_as_token_ids"] = True
        body, _ = self.client.post("/completions", payload)
        # One choice per prompt, but the server is free to reorder them.
        choices = sorted(body["choices"], key=lambda c: c.get("index", 0))
        out = []
        for choice in choices:
            tokens = choice["logprobs"]["tokens"]
            out.append([int(t.split(":", 1)[1]) for t in tokens] if ids else tokens)
        return out

    def encode(self, texts: list[str]) -> list[list[int]]:
        """Token ids for each text, batched into one request where possible."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._memo]
        if missing:
            for encoded, text in zip(self._echo(missing, ids=True), missing):
                self._memo[text] = encoded
            self._save()
        return [self._memo[t] for t in texts]

    def single_token_indexes(self, prefix: str, limit: int = 255) -> list[int]:
        """Ids for ``0..n`` that each add exactly one token after ``prefix``.

        Same contract as upstream's discovery loop, checked against the
        serving tokenizer: an index qualifies when ``prefix + str(i)``
        tokenizes as ``prefix`` plus one further token. Digits are not
        always one token per digit -- this tokenizer has a single token
        for ``12`` -- so the usable range is a property of the model, not
        a constant, and we stop at the first index that does not fit.
        """
        cached = self._indexes.get(prefix)
        if cached and (len(cached["ids"]) >= limit or cached["exhausted"]):
            # A cached probe only answers for a limit it actually reached: a
            # run that asked for 3 indexes must not cap a later one at 3.
            return cached["ids"][:limit]
        base = len(self.encode([prefix])[0])
        candidates = [f"{prefix}{i}" for i in range(limit)]
        encoded = self.encode(candidates)
        ids: list[int] = []
        for tokens in encoded:
            if len(tokens) != base + 1:
                break  # index space exhausted; everything past here is multi-token
            ids.append(tokens[-1])
        if not ids:
            raise ValueError(f"No single-token choice indexes after {prefix!r}")
        self._indexes[prefix] = {"ids": ids, "exhausted": len(ids) < limit}
        self._save()
        return ids
