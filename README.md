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
| Calibration error without labels (excess ECE) | 0.032 | 0.080 | not measured |

Normalized accuracy is 0 for always guessing a dataset's most common label
and 1 for getting everything right, averaged across datasets. On the 28
datasets both can answer, Privatemode Decisions and Jev are statistically
indistinguishable. Jev can't read images, and Laya can't fit 151 options.
The calibration error is what remains after the library's default
temperature, beyond what sampling alone produces (0 is as calibrated as the
test sets can show); Jev's is for its probabilities as returned, and 0.040
if it gets a default temperature fitted the same way.
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
   # payments {'payments': 0.957, 'technical': 0.033, 'complaints': 0.01} 0.82
   ```

The probabilities are calibrated by default: for the models we measured,
the library divides the log probabilities by a pre-configured temperature
before reporting them (see [Calibration](#calibration)).

`confidence` ranges from 0 (probability spread evenly) to 1 (all of it on
one option). It measures how sure the model is, not whether it's right,
and works as a threshold for sending answers to human review; see
[Calibration](#calibration) for how far the probabilities can be trusted.
To include
images, pass `images=` with paths, bytes, Pillow images or data URLs, and
use a vision model such as `glm-flash-latest`.

To build the technique into your own stack with a coding agent, give it
[the blog post](https://privatemode.ai/blog/system-one-from-glm-flash) and
this repository. [AGENTS.md](AGENTS.md) lists what an implementation has to
get right.

## How it works

The prompt numbers the options. The assistant's reply is prefilled with
`answer=`, so the next token the model generates is the number of its
choice. The request restricts generation to those tokens and returns their
log probabilities. The library turns them into probabilities that sum to 1
across your options.

The token IDs for each number depend on the model's tokenizer. The library
gets them from the server, using `/completions` with `echo`, and caches
them for ten minutes per model. The cache expires because an alias such as
`glm-flash-latest` can move to a model with a different tokenizer.

Because the probabilities cover only your options, the model can't answer
"none of these". Add it as an option if you need it.

## Calibration

Raw probabilities from one token are overconfident: on the benchmark,
confidence exceeded accuracy by 15 points on average. So the library
softens them by default, with a temperature that depends on the number of
options, measured per model: GLM-5.3-Flash, Kimi K2.6 and GLM-5.3 (and
their `-latest` aliases). That removes most of the gap without any labels;
other models keep their raw probabilities until they're measured. GLM-5.3
puts only about two thirds of its probability on the options after
`answer=`, so its answers are less reliable than Flash's or Kimi's.

- `SystemOne(..., temperature="sentiment")` uses the temperature for a
  task family (`intent`, `legal`, `moderation`, `nli`, `qa`, `sentiment`,
  `topic`), which fits better if you know what kind of task it is.
  `temperature=1` gives the raw probabilities.
- With labelled answers from a random sample of your inputs, `calibrate()`
  fits your task: a temperature and a bias per option, which corrects a
  model that favours some options and so changes answers (+2.0 points of
  accuracy from 100 labels on the benchmark, +1.0 from 20, +3.0 from 500).
  It also gives two guarantees:

  ```python
  from decisions import calibrate

  calibration = calibrate(answers, labels, coverage=0.9, max_error=0.05)
  calibration.apply(new_answer).choice  # the corrected answer: use this one
  calibration.predict_set(new_answer)   # ['payments'], or several options for a person to pick
  calibration.automate(new_answer)      # True: act on it; errors among these stay at most 5%
  ```

  `predict_set` contains the right option 90% of the time; with
  `per_class=True` that holds for every option, which matters when one is
  rare; an option with fewer than 9 labels (at 90%) is then in every set.
  `automate` acts on answers whose top probability (after the
  correction) clears a fitted threshold, and keeps the error among them at
  most 5% with probability 90% over the choice of labels. That is the top
  probability, not `answer.confidence`, which measures how peaked the whole
  distribution is. A few hundred labels make both reliable, and a guarantee
  costs automation: on the benchmark, a 5% error bound let about a fifth of
  answers through, 10% about a third. `bias=False` fits the temperature
  alone, which never changes an answer and automates a little more (27%
  instead of 23% at a 10% bound with 100 labels). Label a random sample,
  not only escalated cases.
- `evaluate(answers, labels, calibration=...)` reports accuracy, ECE,
  coverage and the error among automated answers, for a fresh audit sample:
  refit with `calibrate()` when they drift.
  [examples/audit_loop.py](examples/audit_loop.py) is such a loop.
- `SystemOne(permutations=k)` asks in k option orders and averages. It keeps
  the one-order default temperature, which measured as good as any other
  choice without labels, and it didn't raise accuracy on the benchmark.

**Which guarantee holds when.** The prediction sets cover the right option
at the stated rate *on average over new inputs drawn like the labelled
ones* (exchangeability); with `per_class=True`, for each option separately.
The error bound on automated answers follows Learn then Test, but its
candidate thresholds come from the labelled answers themselves rather than
a grid fixed in advance, so its 90% over the choice of labels is tested on
the benchmark, not proven. So is fitting the correction on the same labels
as the cutoffs. For a temperature alone at most 1.1% of samples broke the
bound (10% allowed); with a bias, cutoffs and threshold are set
out-of-fold, and at most 1.1% did. Labels collected only from escalated or
disputed cases break all of it.

**Why calibration happens in the client.** The library receives the raw log
probabilities and calibrates on your machine. You refit the temperature,
bias, cutoffs and threshold on your own labels, and the labels never leave
your infrastructure, which matters on a confidential-computing service.
Services that return rounded or already-transformed probabilities only
allow calibration stacked on top of their own transform: Jev rounds to 0.01
and prices the right answer at exactly 0 in 4.3% of the benchmark's
examples, so a temperature can't even be fitted without first patching the
zeros.

**What we tried and dropped.** Each was measured on the benchmark and
didn't beat what the library does:

- Dividing out the answer to a neutral input (contextual calibration): made
  24 of 28 datasets worse, up to 16 points of accuracy; batch calibration
  on unlabelled traffic cost 0.5 points on average. The bias they remove is
  mostly real knowledge or the real class balance.
- Averaging option orders (`permutations`) or a position prior (PriDe):
  −0.3 points, not significant, for 4× the requests. Re-reading only
  uncertain answers in more orders didn't help either, and a temperature
  fitted per number of orders did worse than the one-order default.
- Isotonic regression instead of a temperature: needs about 500 labels to
  catch up. Predicting a task's temperature without labels (Thermometer
  and similar): at most the gap from 0.032 to 0.006 excess ECE, half of
  which 20 labels already close.
- Clustered conformal sets for many options with few labels: no better than
  one cutoff at a few labels per class.
- Correcting answers by known class rates: needs rates as accurate as 100
  labels would give, and hurts when they are off.
- Probability left off the options (option mass): about 99% sits on the
  options, right or wrong, so it says nothing about errors.

The measurements, plots and method are in the benchmark's
[calibration report](https://github.com/edgelesssys/privatemode-decisions-benchmark/tree/main/results/calibration).

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
uv venv --python 3.14 .venv && uv pip install -e '.[dev]'
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
