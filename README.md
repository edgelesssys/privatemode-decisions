# System One with Privatemode AI

A text box in front of a System One engine: type context and
multiple-choice questions, paste images, and see the model's whole
distribution over your choices as bars. Every answer is one forward pass,
read from a single masked token; there is no chain of thought, which is why
several of the prepared questions are traps. Inference runs on
[Privatemode AI](https://privatemode.ai).

```
You are a phone agent answering calls.

Which team should the call go to?
choices: repair, human, sales

Is this a valid request?
choices: yes, no
```

Everything that is not a question is context and goes to the model with
every question. A question is a line followed by a `choices:` line; with
no `choices:` line at all, the last line is a yes/no question. An
option can carry a description: `choices: repair = broken devices, sales =
new orders`. Up to 32 choices per question (the answer is one token).
Images pasted or dropped onto the page are sent as part of the context;
pick a vision model for them.

## Use it in your own code

1. Get a Privatemode API key: <https://portal.privatemode.ai/sign-in/create>.
2. Run the Privatemode proxy. It verifies the deployment's attestation
   report and encrypts every request before it leaves your machine:

   ```sh
   docker run -p 8082:8080 ghcr.io/edgelesssys/privatemode/privatemode-proxy:latest \
     --apiKey "$PRIVATEMODE_API_KEY"
   ```

3. Install the library and ask:

   ```sh
   pip install git+https://github.com/edgelesssys/privatemode-system-one
   ```

   ```python
   from system_one import Choice, OpenAIClient, SystemOne

   engine = SystemOne(OpenAIClient("http://localhost:8082/v1"), "glm-flash-latest")
   result = engine.system_one(
       "A customer writes: I was charged twice for the same transfer on Monday.",
       {
           "team": Choice({
               "payments": "wrong or duplicate charges",
               "technical": "app and login problems",
               "complaints": "escalations and repeat contacts",
           }),
           "urgent": Choice({"yes": None, "no": None}),
       },
   )
   answer = result.answers["team"]
   print(answer.choice, answer.probabilities, answer.confidence)
   # payments {'payments': 0.999, 'technical': 0.0, 'complaints': 0.0} 1.0
   ```

Each question costs one forward pass and one output token, and the answer
comes with the whole distribution over its options. Threshold on
`confidence` (1 minus normalized entropy) to decide which answers a human
should see.

To build it with a coding agent, give Claude Code, Codex or the tool of
your choice the post that explains the technique,
<https://privatemode.ai/blog/system-one-from-glm-flash>, together with
this repository. [AGENTS.md](AGENTS.md) lists what an implementation has
to get right.

## Run the web app

```sh
uv venv --python 3.12 .venv && uv pip install -e '.[dev]'
cp .env.example .env            # proxy URL and Privatemode API key
./run.sh                        # http://127.0.0.1:8600
.venv/bin/pytest -q tests       # the parser
sh deploy.sh                    # docker compose on a host, behind your reverse proxy
```

The backend talks to a Privatemode proxy (`privatemode-proxy` container,
OpenAI-compatible, `/completions` with logprobs); the key is passed per
request. `system_one/` is the System One library against a remote
OpenAI-compatible endpoint: `SystemOne.system_one(context, {key:
Choice(question, options)}, images=...)` is the whole integration, in
`app/main.py`.

## How it works

Prefill `choice_index:` into the assistant turn, mask sampling to the token
ids that are valid option indexes, read the answer and the full distribution
out of that one logit row. The server's own tokenizer is the oracle for
which ids those are (`/completions` with `echo` and `logprobs`), cached per
model. Masked probabilities renormalize over the options and carry no
"none of these" signal; add that as a choice if you want it.

## Layout

```
system_one/      the library: client, token oracle, prompt building, image handling
app/parse.py     the text format -> context + questions, with errors that explain the format
app/samples.py   the prepared questions and why each is there
app/main.py      FastAPI: /api/ask, /api/parse, /api/samples, /api/models
web/index.html   the page: editor, image thumbnails, sample chips, bar charts
tests/           parser tests (the model is not needed)
```
