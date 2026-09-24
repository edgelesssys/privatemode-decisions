# Privatemode Decisions

Ask an LLM to choose from a fixed set of options and get its choice plus a
probability for each option. Each decision takes one forward pass and
produces one token. No fine-tuning, no free text to parse. It works like
TypeSafe's Jev, a System One model, but runs on GLM-5.3-Flash.

- **Speed:** about 150 ms per decision, network included.
- **Cost:** almost entirely input tokens, since the answer is one token.
- **Context:** up to 1M tokens per decision.
- **Images:** include screenshots, scans or photos alongside the text.
- **Options:** up to 191 per question, with optional descriptions.
- **Confidentiality:** on [Privatemode AI](https://privatemode.ai), your
  data stays encrypted even while it's processed, and the deployment's
  attestation is verified before anything is sent.
- **Portability:** works with any vLLM-backed, OpenAI-compatible server,
  without the confidentiality guarantees.
- **Model choice:** the library gets the token IDs it needs from the
  model's tokenizer, so you can switch models without changing code.

The repository contains `decisions/`, a Python library
and a sample web app that shows the full probability distribution for each answer.
See our
[blog post](https://privatemode.ai/blog/system-one-from-glm-flash) for a high level explanation.

## Benchmark

We compared Privatemode Decisions with TypeSafe's Jev and Convai's Laya on
29 public labelled datasets in English and German, with 2 to 151 options
and up to 1,000 examples each.

| | Privatemode Decisions | Jev | Laya |
|---|---|---|---|
| Datasets it can answer | 29 | 28 | 27 |
| Normalized accuracy | 0.585 | 0.574 | 0.422 |
| Median latency, from Germany | 152 ms | 251 ms | runs locally |
| EUR per 1,000 decisions | 0.062 | 0.016 | runs locally |

Normalized accuracy is 0 for always guessing a dataset's most common label
and 1 for getting everything right, averaged across datasets. On the 28
datasets both can answer, Privatemode Decisions and Jev are statistically
indistinguishable. Jev can't read images, and Laya can't fit 151 options.
Full results and methodology are in
[privatemode-decisions-benchmark](https://github.com/edgelesssys/privatemode-decisions-benchmark).

## Quickstart

1. Get a Privatemode API key: <https://portal.privatemode.ai/sign-in/create>.
2. Start the Privatemode proxy on the machine sending the requests. It
   verifies the deployment's attestation and encrypts each request before
   it leaves your machine. Bind it to localhost, because anyone who can
   reach the proxy can use your key.

   ```sh
   docker run -p 127.0.0.1:8080:8080 ghcr.io/edgelesssys/privatemode/privatemode-proxy:latest \
     --apiKey "$PRIVATEMODE_API_KEY"
   ```

3. Install the library. Add the `[images]` extra for automatic image
   resizing.

   ```sh
   pip install git+https://github.com/edgelesssys/privatemode-decisions
   ```

4. Ask a question:

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
           "urgent": Choice({"yes": None, "no": None}, instructions="Is this urgent?"),
       },
   )
   answer = result.answers["team"]
   print(answer.choice, answer.probabilities, answer.confidence)
   # payments {'payments': 0.997, 'technical': 0.002, 'complaints': 0.0} 0.98
   ```

`confidence` ranges from 0 (probability spread evenly) to 1 (all of it on
one option). It measures how sure the model is, not whether it's right,
and works as a threshold for sending answers to human review. To include
images, pass `images=` with paths, bytes, Pillow images or data URLs, and
use a vision model such as `glm-flash-latest`.

To build the technique into your own stack with a coding agent, give it
[the blog post](https://privatemode.ai/blog/system-one-from-glm-flash) and
this repository. [AGENTS.md](AGENTS.md) lists what an implementation has to
get right.

## How it works

The prompt numbers the options. The assistant's reply is prefilled with
`answer:`, so the next token the model generates is the number of its
choice. The request restricts generation to those tokens and returns their
log probabilities. The library turns them into probabilities that sum to 1
across your options.

The token IDs for each number depend on the model's tokenizer. The library
gets them from the server, using `/completions` with `echo`, and caches
them for ten minutes per model. The cache expires because an alias such as
`glm-flash-latest` can move to a model with a different tokenizer.

Because the probabilities cover only your options, the model can't answer
"none of these". Add it as an option if you need it.

## The web app

The web app lets you try the library without writing code. Enter context
and one or more questions, and it shows the probability of every option as
a bar chart. Some of the prepared examples are trick questions that show
what happens when a model has to answer without thinking first.

The format is plain text:

- A question is a line followed by a `choices:` line.
- Options can have descriptions: `choices: repair = broken devices, sales = new orders`.
- Every other line is context and is sent with every question.
- Without any `choices:` line, the last line becomes a yes/no question.
- Pasted or dropped images are sent as context and need a vision model.
- The demo allows up to 10 questions and 32 options per question. These are
  demo limits, not library limits.

For example:

```
You are a phone agent answering calls.

Which team should the call go to?
choices: repair, human, sales

Is this a valid request?
choices: yes, no
```

To run it:

```sh
uv venv --python 3.12 .venv && uv pip install -e '.[dev]'
cp .env.example .env         # proxy URL and Privatemode API key
./run.sh                     # http://127.0.0.1:8600
.venv/bin/pytest -q tests    # no model needed
sh deploy.sh                 # docker compose on a host, behind your reverse proxy
```

`./run.sh` expects the proxy from the Quickstart on `localhost:8080`.
`docker compose` starts its own proxy, which only the app can reach. The
app is a showcase and sends every visitor's questions with the single key
in `.env`. If you make it public, add authentication or rate limiting.

## Layout

```
decisions/       the library: client, token oracle, prompt building, images
app/parse.py     parses the text box into context and questions, with helpful errors
app/samples.py   the prepared examples and why each one is there
app/main.py      FastAPI: /api/ask, /api/parse, /api/samples, /api/models
web/index.html   the page: editor, image thumbnails, example chips, bar charts
tests/           parser, token oracle, batching, decoding and API tests against fakes
```
