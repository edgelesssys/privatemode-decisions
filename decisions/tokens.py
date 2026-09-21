"""Finding single-token option labels without a local tokenizer.

The option numbers have to be single tokens, and which ones are depends
on the tokenizer. There is no local tokenizer to ask: the model lives
inside a confidential-computing enclave and all we have is an
OpenAI-compatible API. So we use the server's own tokenizer as an oracle.

vLLM's ``/completions`` accepts ``echo`` (return the prompt) together
with ``logprobs`` (return per-token detail) and the vLLM extra
``return_tokens_as_token_ids``. Echoing a zero-token completion is a
prefill-only round trip that hands back the exact tokenization of the
prompt -- ids with the flag, text without it. That is a complete
``encode()``, and it is authoritative for the model actually serving,
which a locally downloaded tokenizer is only ever assumed to be.

Results are kept in memory for ``max_age`` seconds and then probed again.
A model name is not a tokenizer version: when an alias such as
``glm-flash-latest`` moves to a model with another tokenizer, cached ids
would point at the wrong tokens, and the mask would still force one of them
-- every answer would look normal and be wrong. There is no disk cache for
the same reason; checking one would cost the same request as probing again.
"""

from __future__ import annotations

import threading
import time

from .client import OpenAIClient

#: Seconds a probe is trusted before the next use probes again.
MAX_AGE = 600.0


class TokenOracle:
    """Tokenizes through the serving endpoint and remembers it for a while."""

    def __init__(self, client: OpenAIClient, model: str, *, max_age: float = MAX_AGE) -> None:
        self.client, self.model = client, model
        self.max_age = max_age
        self._memo: dict[str, list[int]] = {}
        self._indexes: dict[str, dict] = {}
        self._since = time.monotonic()
        # Concurrent first questions would otherwise probe the same thing twice.
        self._lock = threading.RLock()

    def _expire(self) -> None:
        if time.monotonic() - self._since > self.max_age:
            self._memo.clear()
            self._indexes.clear()
            self._since = time.monotonic()

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
        with self._lock:
            self._expire()
            missing = [t for t in dict.fromkeys(texts) if t not in self._memo]
            if missing:
                for encoded, text in zip(self._echo(missing, ids=True), missing):
                    self._memo[text] = encoded
            return [self._memo[t] for t in texts]

    def single_token_indexes(self, prefix: str, limit: int = 255) -> list[int]:
        """Ids for ``0..n`` that each add exactly one token after ``prefix``.

        Checked against the serving tokenizer: an index qualifies when
        ``prefix + str(i)`` tokenizes as ``prefix`` plus one further token. Digits are not
        always one token per digit -- GLM-5.3-Flash has single tokens for
        0 to 190 -- so the usable range is a property of the model, not a
        constant, and we stop at the first index that does not fit.
        """
        with self._lock:
            self._expire()
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
            return ids
