# AGENTS.md

Instructions for coding agents. Two kinds of work happen against this
repository: building the technique into someone else's code, with
`decisions/` as the reference implementation, and changing this repository
itself. Both need the first two sections.

## What this is

A decision: a piece of state, a question, and a fixed set of
named options go in; the chosen option and a probability for every option
come out, from one forward pass of an ordinary LLM. No fine-tuning, no
parsing of free text. It works because the prompt puts the whole answer at
one token position and the request reads the distribution there instead of
sampling a sentence.

It needs a vLLM-backed, OpenAI-compatible endpoint, because it relies on
vLLM's extra request fields. On Privatemode that endpoint is the
`privatemode-proxy` container, which verifies the deployment's attestation
and encrypts the traffic. Send requests to the proxy, never around it.

## What an implementation has to get right

Each point is a way a port silently returns plausible but wrong
probabilities. `decisions/inference.py` and `decisions/tokens.py` are the
reference for all of them.

1. **Number the options, in order.** The user message is the preamble
   (`PREAMBLE`) followed by JSON with `state`, `question` and `options`,
   each option as `{"number": i, "label": ..., "description": ...}`. The
   number is the option's position. Keep the mapping from number back to
   option name; everything downstream depends on it.
2. **Prefill the answer.** Append an assistant message with content
   `answer:` and send `continue_final_message: true` and
   `add_generation_prompt: false`. Without both, the model opens a new turn
   and the next token is formatting, not the index.
3. **Find the index token ids with the serving tokenizer.** An index `i`
   qualifies when `answer:` + `str(i)` tokenizes as the tokens of
   `answer:` plus exactly one more; that last token's id is the index
   id. Tokenize the prefixed string, not the bare digits, because
   tokenization depends on what precedes it. Stop at the first index that
   doesn't fit. Do not assume one digit per token: GLM-5.3-Flash has single
   tokens for 0 to 190. The oracle is `/completions` with `echo: true`,
   `logprobs: 0`, `max_tokens: 0` and `return_tokens_as_token_ids: true`,
   which returns the exact tokenization by the model that is serving. Cache
   the result per model; a model name is not a tokenizer version, so drop
   the cache when the deployment changes. Where `/completions` isn't
   reachable, pin the ids discovered this way and refuse unknown models
   rather than guessing.
4. **Mask to the options.** `allowed_token_ids` = the index ids of this
   question's options, `max_tokens: 1`, `temperature: 0`. The one generated
   token is then necessarily an option index.
5. **Read the option logprobs with `logprob_token_ids`**, set to the same
   ids. `top_logprobs` alone is not enough: it reports the distribution
   before the mask, where formatting tokens such as a leading space take
   the top slots and push real options off the list, so they read as zero.
6. **Map by id, not by string.** Send `return_tokens_as_token_ids: true` and
   match the `token_id:<n>` entries against the allowed ids. Decoded token
   strings bring whitespace and casing ambiguity.
7. **Renormalize over the options.** Exponentiate the option logprobs and
   divide by their sum. The result carries no "none of these" signal; if the
   caller needs one, it has to be an option.
8. **Report confidence as what it is.** The library returns
   `1 - entropy / log(n)`. That measures how peaked the distribution is, not
   whether the answer is right: a model can be confidently wrong, which is
   what trick questions show.

## Limits

- The number of options is capped by the single-token indexes the model
  has (191 on GLM-5.3-Flash); `SystemOne.system_one` raises `ValueError`
  when they run out.
- Privatemode reports at most 128 ids per response: a longer
  `logprob_token_ids` is rejected with HTTP 400, while `allowed_token_ids`
  takes all 191. With more options, keep the full mask in every request and
  split only the ids to read into batches of 128, then merge before
  renormalizing (`batches`, `max_logprob_ids`). The reported logprobs are
  raw, taken before the mask, so batches under the same mask are slices of
  one distribution. Measured: the same id reads identically in two batches,
  and merged results differ from a single read by no more than two
  identical single reads differ from each other.
- Option order is a prior of its own: a model can prefer an index regardless
  of the question. `permutations=k` asks each question in `k` rotated orders
  and averages the probabilities. Rotations, not shuffles, so scales keep
  their order and results stay deterministic.
- Put what is shared across questions first (preamble, state, images) so
  vLLM's prefix cache reuses it. `mode="staged"` sends one question first to
  seat that prefix, then the rest in parallel.
- The client holds at most `MAX_IN_FLIGHT` (9) requests in flight per
  process. The proxy is shared; raise the cap deliberately, with
  `decisions.client.set_max_in_flight`, not by working around it.

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
them; use a vision model (`glm-flash-latest`) for them. `app/main.py` is a
complete integration.

## Working in this repository

```sh
uv venv --python 3.12 .venv && uv pip install -e '.[dev]'
cp .env.example .env        # proxy URL and Privatemode API key
.venv/bin/pytest -q tests   # what CI runs, with a docker build
./run.sh                    # the demo on http://127.0.0.1:8600
```

- `decisions/` has no required dependencies beyond the standard library
  (Pillow only for resizing images). Keep it that way; no torch, no
  transformers, no HTTP client packages.
- `decisions/types.py` mirrors the shapes of TypeSafe's `typesafe_sdk`
  (`Choice`, `ChoiceAnswer`, `SystemOneResponse`, `Usage`). Don't change
  them without a reason; callers port code across.
- The tests cover the text-box parser, the token oracle, the request
  batching and decoding, and the API (against fakes) and need no model. A change to the
  request or the decoding needs a check against a running proxy; say so in
  the pull request if you couldn't run one.
- Never commit `.env` or an API key. The demo reads one key from `.env`
  and uses it for every visitor's questions.
- The demo caps a question at 32 options (`app/parse.py`); that's a limit of
  the demo, not of the library.
