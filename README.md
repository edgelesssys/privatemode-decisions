# Privatemode Decisions

Typed decisions from an ordinary LLM: pass in some state and a fixed set of
named options, get back the chosen option and a probability for every
option. One forward pass, one output token, no fine-tuning and no parsing
of free text. It works like a System One model such as TypeSafe's Jev, built
on GLM-5.3-Flash.

- **Fast:** a typical decision takes about 150 ms, network included.
- **You pay only for input:** the answer is a single output token.
- **Long context:** up to 1M tokens of state per decision.
- **Images:** pass screenshots, scans or photos along with the text.
- **Up to 191 options** per question, each with an optional description.
- **Confidential:** on [Privatemode AI](https://privatemode.ai), data stays
  encrypted during processing with verified deployment attestation.
- **Any vLLM endpoint:** it works against any vLLM-backed, OpenAI-compatible
  server, without the confidentiality guarantees.
- **Switch models freely:** the option token ids are read from the serving
  model's tokenizer, so there is nothing model-specific to ship.

This repository has two parts: `decisions/`, a dependency-free Python
library, and a web app that shows the model's whole distribution as bars.
The [blog post](https://privatemode.ai/blog/system-one-from-glm-flash)
explains the technique and benchmarks it against Jev on 29 datasets.

## Quickstart

1. Get a Privatemode API key: <https://portal.privatemode.ai/sign-in/create>.
2. Run the Privatemode proxy on the machine that sends the requests. It
   verifies the deployment's attestation and encrypts every request before
   it leaves your machine. Bind it to localhost: anyone who reaches it can
   use your key.

   ```sh
   docker run -p 127.0.0.1:8080:8080 ghcr.io/edgelesssys/privatemode/privatemode-proxy:latest \
     --apiKey "$PRIVATEMODE_API_KEY"
   ```

3. Install the library and ask:

   ```sh
   pip install git+https://github.com/edgelesssys/privatemode-decisions
   # add [images] for Pillow, to resize images (image_max_side)
   ```

   ```python
   from decisions import Choice, OpenAIClient, SystemOne

   engine = SystemOne(OpenAIClient("http://localhost:8080/v1"), "glm-flash-latest")
   result = engine.system_one(
       "A customer writes: I was charged twice for the same transfer on Monday.",
       {
           "team": Choice({
               "payments": "wrong or duplicate charges",
               "technical": "app and login problems",
               "complaints": "escalations and repeat contacts",
           }, instructions="Which team should handle this?"),
           "urgent": Choice({"yes": None, "no": None}),
       },
   )
   answer = result.answers["team"]
   print(answer.choice, answer.probabilities, answer.confidence)
   # payments {'payments': 0.999, 'technical': 0.0, 'complaints': 0.0} 1.0
   ```

`confidence` is 1 minus the normalized entropy: how peaked the distribution
is, not whether the answer is right. Threshold on it to decide which answers
a human should see. Pass `images=` (paths, bytes, Pillow images or data URLs)
with a vision model such as `glm-flash-latest`.

To port the technique into your own stack with a coding agent, give it
[the blog post](https://privatemode.ai/blog/system-one-from-glm-flash) and
this repository. [AGENTS.md](AGENTS.md) lists what an implementation has to
get right.

## How it works

The prompt numbers the options and prefills `answer:` into the
assistant turn, so the whole answer sits at one token position. The request
masks sampling to the token ids of the option indexes and reads their
logprobs, then renormalizes over the options. The serving model's own
tokenizer (`/completions` with `echo`) says which ids those are; the result
is cached per model for ten minutes, since an alias like `glm-flash-latest`
can move to a model with another tokenizer.

Because probabilities are renormalized over your options, there is no "none
of these" signal. Add it as an option if you need it.

## The web app

A text box in front of the library: type context and questions, paste
images, and see the distribution. Several prepared questions are deliberate
traps, since there is no reasoning step.

```
You are a phone agent answering calls.

Which team should the call go to?
choices: repair, human, sales

Is this a valid request?
choices: yes, no
```

- Every line that isn't a question is context, sent with every question.
- A question is a line followed by `choices:`. Without any `choices:` line,
  the last line is a yes/no question.
- Options can carry a description: `choices: repair = broken devices, sales = new orders`.
- Up to 10 questions and 32 options per question (limits of the demo, not
  the library).
- Pasted or dropped images join the context; pick a vision model for them.

```sh
uv venv --python 3.12 .venv && uv pip install -e '.[dev]'
cp .env.example .env         # proxy URL and Privatemode API key
./run.sh                     # http://127.0.0.1:8600
.venv/bin/pytest -q tests    # no model needed
sh deploy.sh                 # docker compose on a host, behind your reverse proxy
```

`./run.sh` expects the proxy from the quickstart on `localhost:8080`;
`docker compose` starts its own, reachable only by the app. The app is a
showcase: it uses the single key from `.env` for every visitor, so put
authentication or rate limiting in front of it if you expose it.

## Layout

```
decisions/       the library: client, token oracle, prompt building, images
app/parse.py     text format -> context + questions, with errors that explain the format
app/samples.py   the prepared questions and why each is there
app/main.py      FastAPI: /api/ask, /api/parse, /api/samples, /api/models
web/index.html   the page: editor, image thumbnails, sample chips, bar charts
tests/           parser, token oracle, batching, decoding and API tests against fakes
```
