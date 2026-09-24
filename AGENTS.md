# AGENTS.md

Instructions for coding agents, whether you're building the technique into
other code (with `decisions/` as the reference) or changing this
repository. Both need the first two sections.

## What this is

A piece of state, a question and a fixed set of named options go in; the
chosen option and a probability for every option come out, from one
forward pass of an ordinary LLM. The prompt puts the whole answer at one
token position, and the request reads the distribution there instead of
sampling text.

It needs a vLLM-backed, OpenAI-compatible endpoint, because it uses vLLM's
extra request fields. On Privatemode that endpoint is the
`privatemode-proxy` container, which verifies the deployment's attestation
and encrypts the traffic. Run it on the same machine or network namespace
as the caller, since everything before it is plaintext, and never send
requests around it.

## What an implementation has to get right

Each point is a way a port silently returns plausible but wrong
probabilities. `decisions/inference.py` and `decisions/tokens.py` are the
reference.

1. **Number the options in order.** The user message is `PREAMBLE`
   followed by JSON with `state`, `question` and `options`, each option as
   `{"number": i, "label": ..., "description": ...}`, where `i` is its
   position. Keep the mapping from number back to option.
2. **Prefill the answer.** Append an assistant message `answer:` and send
   `continue_final_message: true` and `add_generation_prompt: false`.
   Without both, the model opens a new turn and the next token is
   formatting, not a number.
3. **Get the number token IDs from the serving tokenizer.** `i` qualifies
   when `answer:` + `str(i)` tokenizes as the tokens of `answer:` plus
   exactly one more; that token is the ID. Tokenize the prefixed string,
   not bare digits, and stop at the first `i` that doesn't fit. Don't assume
   one digit per token: GLM-5.3-Flash has single tokens for 0 to 190. Use
   `/completions` with `echo: true`, `logprobs: 0`, `max_tokens: 0` and
   `return_tokens_as_token_ids: true`. Cache per model, but let the cache
   expire: a model alias can move to another tokenizer. Without
   `/completions`, pin IDs found this way and refuse unknown models.
4. **Mask to the options.** `allowed_token_ids` = this question's IDs,
   `max_tokens: 1`, `temperature: 0`.
5. **Read the logprobs with `logprob_token_ids`**, set to the same IDs.
   `top_logprobs` alone isn't enough: it reports the distribution before
   the mask, where formatting tokens push real options off the list, so
   they read as zero.
6. **Map by ID, not by string.** Send `return_tokens_as_token_ids: true`
   and match the `token_id:<n>` entries. Decoded strings bring whitespace
   and casing ambiguity.
7. **Renormalize over the options.** Exponentiate and divide by the sum.
   There is no "none of these" signal unless it's an option.
8. **Report confidence as what it is.** `1 - entropy / log(n)` measures
   how peaked the distribution is, not whether the answer is right.

## Limits

- Options are capped by the model's single-token numbers (191 on
  GLM-5.3-Flash); `system_one` raises `ValueError` beyond that.
- Privatemode reports at most 128 IDs per response and rejects a longer
  `logprob_token_ids` with HTTP 400; `allowed_token_ids` takes all 191.
  With more options, send the full mask in every request, read the IDs in
  batches of 128, and merge before renormalizing (`batches`,
  `max_logprob_ids`). The logprobs are taken before the mask, so the
  batches are slices of one distribution; measured, merging adds no error.
- A model can prefer a position regardless of the question.
  `permutations=k` asks in `k` rotated orders and averages. Rotations keep
  scales in order and results deterministic.
- Put what's shared across questions first (preamble, state, images) so
  vLLM's prefix cache reuses it. `mode="staged"` sends one question first
  to seat the prefix, then the rest in parallel.
- At most `MAX_IN_FLIGHT` (9) requests are in flight per process. Raise it
  deliberately with `decisions.client.set_max_in_flight`.

## Using the library

```python
from decisions import Choice, SystemOne

engine = SystemOne.from_env("glm-flash-latest")  # DECISIONS_BASE_URL, DECISIONS_API_KEY
result = engine.system_one(
    {"ticket": "My washing machine stopped mid-cycle."},
    {"team": Choice({"repair": "broken devices", "sales": "new orders", "human": None},
                    instructions="Which team should handle this ticket?")},
)
answer = result.answers["team"]
answer.choice, answer.probabilities, answer.confidence
```

`images=` takes a path, bytes, a Pillow image, a data URL, or a list of
them, and needs a vision model such as `glm-flash-latest`. Strings that
aren't URLs are read as file paths, so don't pass user input there.
`app/main.py` is a complete integration.

## Working in this repository

```sh
uv venv --python 3.14 .venv && uv pip install -e '.[dev]'
cp .env.example .env        # proxy URL and Privatemode API key
.venv/bin/pytest -q tests   # CI runs this plus a docker build
./run.sh                    # the demo on http://127.0.0.1:8600
```

- `decisions/` needs only the standard library (Pillow optionally, for
  resizing images). Keep it that way: no torch, no transformers, no HTTP
  client packages.
- `SystemOne`, `system_one()` and the types in `decisions/types.py` mirror
  TypeSafe's SDK so Jev code ports over. Don't rename them.
- The tests need no model. A change to the request or decoding needs a
  check against a running proxy, and a change to the prompt needs a
  benchmark run
  ([privatemode-decisions-benchmark](https://github.com/edgelesssys/privatemode-decisions-benchmark)).
  Say so in the pull request if you couldn't run them.
- Never commit `.env` or an API key. The demo uses the one key in `.env`
  for every visitor, so it caps a request at 10 questions and 32 options
  (`app/parse.py`) and accepts only `application/json` bodies. These are
  limits of the demo, not the library.
