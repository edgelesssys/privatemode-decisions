# Privatemode Decisions

Privatemode Decisions makes an LLM choose from a fixed set of options:
you get its choice plus a calibrated probability for each option, from
one forward pass and one token. No fine-tuning, no free text to parse.
It works like TypeSafe's Jev but runs on GLM-5.3-Flash.

- **Fast and cheap:** about 150 ms per decision, network included; you
  pay almost only for input tokens.
- **Inputs:** up to 1M tokens of context, images, up to 191 options per question.
- **Calibrated:** by default, and fitted to your task from a few labels.
- **Confidential:** on [Privatemode AI](https://privatemode.ai), data stays
  encrypted while it's processed, and the deployment's attestation is
  verified before anything is sent.
- **Portable:** any vLLM-backed, OpenAI-compatible server works (without
  the confidentiality guarantees).

The repository holds `decisions/`, a Python library, and a web app that
shows the probability of every option. The
[blog post](https://privatemode.ai/blog/system-one-from-glm-flash) explains
the idea.

## JevBench

Public-item accuracy of the System One models in JevBench v1.4.2 (one
pass, no thinking), down to Jev:

| | public items |
|---|---:|
| Gemma 4 31B IT (Autoloops, Kushal Patil) | 0.928¹ |
| NInfer (Qwen3.8-Flash-Next) | 0.896 |
| JevOne (Qwen3.6-35B-A3B) | 0.896 |
| **Privatemode Decisions (GLM-5.3-Flash)**² | **0.894** |
| swanOne (Qwen3.8-Flash-Next) | 0.887 |
| Jev-Omni (Gemma 4 12B) | 0.887 |
| Cygnet (Gemma 4 12B) | 0.879 |
| reflex-27b (Qwen3.8-27B) | 0.870 |
| Jev 1.13.0 (TypeSafe) | 0.866 |
| SimpleJev (Qwen3.8-27B) | 0.866 |
| Instinct (ZooWork, Qwen3.8-27B) | 0.866 |

¹ 350 of 377 items; every other row is out of the 231 public items.

² Our own run (mean of two), not an official entry. The official runs
also cover the sealed items, on which the systems above score 47 to 57
points lower; we can't run those.

## Benchmark

We also compared 29 public labelled datasets in English and German,
2 to 151 options, the same held-out examples for every system (up
to 500 per dataset):

| | Privatemode Decisions | Jev |
|---|---|---|
| Datasets it can answer | 29 | 28 |
| Mean accuracy, the 28 datasets Jev answers | **0.798** | 0.775 |
| Median latency, from Germany | about 150 ms | 251 ms |
| EUR per 1,000 decisions | 0.095 | 0.016 |
| Calibration error without labels (excess ECE) | **0.032** | 0.080 |

On the same examples Privatemode Decisions is ahead of Jev on 16
datasets, tied on 8 and behind on 3 (Wilcoxon p = 0.001). Jev can't read
images. Our calibration error is the state-first prompt's, with the
default temperature formula fitted without the dataset at hand. The
shipped temperatures, fitted on all datasets, give 0.016 with the current
prompt against 0.023 with state first. Jev's is as returned (0.040 with a
default temperature of its own). Methodology and full results:
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

`confidence` is 0 for probability spread evenly and 1 for all of it on one
option: how sure the model is, not whether it's right. `result.model` is
the model that answered (`glm-flash-latest` comes back as
`glm-5.3-flash`); log it with each decision. For images, pass `images=`
(paths, bytes, Pillow images or data URLs) with a vision model such as
`glm-flash-latest`. To build the technique into your own stack, give a
coding agent the blog post and this repository; [AGENTS.md](AGENTS.md)
lists what an implementation has to get right.

## How it works

The prompt numbers the options and states the question before and after
the state, so the model reads the state knowing what it is asked (+1.6
points on the benchmark). The reply is prefilled with `answer=`; the
request allows only the option numbers and returns their log
probabilities, which the library normalizes over your options. The token
IDs come from the serving tokenizer and are cached for ten minutes per
model. The model can't answer "none of these" unless it's an option.

**Several questions about one state.** By default each request leads with
its own question (`optimize="accuracy"`) and the requests go out in
parallel. `optimize="cost"` leads every request with all of the call's
questions, so they share a prefix (per option order) Privatemode can
cache (from about 2,300 tokens), and seats it with one request first. It
pays off only on states that long; on shorter ones it sends N question
blocks per request for nothing. With five questions per state (timings
with `mode="staged"`):

| `optimize` | accuracy vs state first | short states (p50) | 2,000-token states (p50, cached) |
|---|---|---|---|
| `"accuracy"` (default) | **+2.2 points** | **409 ms** | 935 ms, 0% |
| `"cost"` | +0.9 points | 530 ms | **670 ms, 59%** |

With one `history=` string for the call (a record that grows across
calls) every question leads, whatever `optimize` says, so the record is
shared by the call's requests and by later calls with the same
questions; that is the `"cost"` layout and its accuracy. A history per
question (a mapping) is its request's alone, so there its own question
leads.

## Calibration

Raw one-token probabilities are overconfident (by 15 points on the
benchmark). The library divides the log probabilities by a temperature
from the number of options, measured for GLM-5.3-Flash, Kimi K2.6 and
GLM-5.3 and their `-latest` aliases; other models stay raw. That removes
most of the gap without labels, and held up on five untouched tasks.
GLM-5.3 puts only about two thirds of its probability on the options, so
its answers are less reliable than Flash's or Kimi's.
`temperature="sentiment"` (or another task family) uses that family's
value; `temperature=1` gives raw values.

With labels from a random sample of your inputs (not only escalated or
disputed cases, which break all of the below), `calibrate()` fits your
task:

```python
from decisions import calibrate

calibration = calibrate(answers, labels, coverage=0.9, max_error=0.05)
calibration.apply(new_answer).choice  # the corrected answer: use this one
calibration.predict_set(new_answer)   # ['payments'], or several options for a person to pick
calibration.automate(new_answer)      # True: act on it; errors among these stay at most 5%
```

It fits a temperature and a bias per option (+2.0 points of accuracy from
100 labels on the benchmark; `bias=False` for the temperature alone).
Prediction sets hold the right option 90% of the time, on average over
inputs drawn like the labelled ones (`per_class=True`: for each option).
Automation keeps the error among automated answers at most 5% with
probability 90% over the labels (Learn then Test); its thresholds come
from the labels themselves, so that is tested on the benchmark, not
proven. `evaluate()` checks all of it on a fresh sample, and
[examples/audit_loop.py](examples/audit_loop.py) refits on drift. It all
runs in the client, so your labels stay with you. The
[calibration report](https://github.com/edgelesssys/privatemode-decisions-benchmark/tree/main/results/calibration) has the numbers and
what was tried and dropped.

## The web app

Enter context and one or more questions; the app shows every option's
probability as a bar chart. A question is a line followed by a `choices:`
line (`choices: repair = broken devices, sales = new orders` for
descriptions); every other line is context; without a `choices:` line the
last line becomes a yes/no question; pasted images are sent as context.
The demo allows 10 questions and 32 options per question.

```sh
uv venv --python 3.14 .venv && uv pip install -e '.[dev]'
cp .env.example .env         # proxy URL and Privatemode API key
./run.sh                     # http://127.0.0.1:8600, with the proxy on localhost:8080
.venv/bin/pytest -q tests    # no model needed
sh deploy.sh                 # docker compose with its own proxy, behind your reverse proxy
```

The app sends every visitor's questions with the single key in `.env`; if
you make it public, add authentication or rate limiting.

## Layout

```
decisions/       the library: client, token oracle, prompt building, calibration, images
app/             the web app: parser, prepared examples, FastAPI endpoints
web/index.html   the page: editor, image thumbnails, example chips, bar charts
examples/        the audit loop
tests/           everything above against fakes
```

## Changelog

### Accuracy: the question comes first ([#3](https://github.com/edgelesssys/privatemode-decisions/pull/3))

- The prompt states the question before the state as well as after it:
  +1.6 points on the benchmark and 0.885 → 0.894 on JevBench, for 1.6×
  the prompt tokens; MMLU-Pro drops from 63.1% to 61.9%. The default
  temperatures still fit.
- `optimize="accuracy"` (default) leads each request with its own question;
  `"cost"` with all of the call's, for a cacheable prefix. One history for
  all questions always gets the whole block. `mode` defaults to parallel
  unless the requests share more than the preamble. The state-first
  prompt is gone; `ae35442` reproduces its results.

### Calibration ([#2](https://github.com/edgelesssys/privatemode-decisions/pull/2))

- Calibrated by default, with a temperature per measured model and number
  of options (excess ECE 0.129 → 0.032 without labels).
- `calibrate()` fits a temperature and a bias per option (+2.0 points from
  100 labels), with prediction sets and an error bound for automated
  answers; `evaluate()` and the audit loop check them.
- The prefill is `answer=`; answers carry `option_mass`, the `temperature`
  and the `model` they came from.
