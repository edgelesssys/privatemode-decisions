"""The demo backend: parse the text box, ask the model, return the distributions.

    DECISIONS_BASE_URL=http://localhost:8080/v1 DECISIONS_API_KEY=... uvicorn app.main:app

Both variables are also read from a ``.env`` file in the repository root.
The library is ``decisions/`` in this repository, used as it is.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from .parse import ParseError, parse
from .samples import SAMPLES

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

# Models the proxy serves that answer a masked single token sensibly. The
# first is the default: it is the one that also takes images.
MODELS = [
    {"id": "glm-flash-latest", "label": "GLM flash (vision)", "vision": True},
    {"id": "glm-latest", "label": "GLM", "vision": False},
    {"id": "gpt-oss-120b", "label": "gpt-oss 120B", "vision": False},
    {"id": "kimi-latest", "label": "Kimi (vision)", "vision": True},
]
MAX_IMAGES = 6
MAX_TEXT = 20_000                  # characters in the text box
MAX_IMAGE_URL = 8 * 1024 * 1024    # characters in one image's data URL
MAX_BODY = MAX_TEXT * 4 + MAX_IMAGES * MAX_IMAGE_URL + 4096
NO_CONTEXT = "Answer from your own judgment; there is no further context."

log = logging.getLogger(__name__)


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


load_env()

app = FastAPI(title="Privatemode Decisions")
_engines: dict[str, object] = {}
_lock = threading.Lock()


def engine(model: str):
    """One SystemOne per model, so the token oracle's cache is reused."""
    from decisions import SystemOne
    with _lock:
        if model not in _engines:
            _engines[model] = SystemOne.from_env(model)
        return _engines[model]


async def read_body(request: Request) -> dict | JSONResponse:
    """The JSON object a POST carries, or the error response to return."""
    # A JSON content type forces a CORS preflight, so other sites cannot
    # make a visitor's browser post here with a plain form.
    if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
        return JSONResponse({"error": "the body must be sent as application/json"}, status_code=415)
    try:
        if int(request.headers.get("content-length") or 0) > MAX_BODY:
            return JSONResponse({"error": "request too large"}, status_code=413)
    except ValueError:
        return JSONResponse({"error": "bad Content-Length"}, status_code=400)
    raw = await request.body()
    if len(raw) > MAX_BODY:
        return JSONResponse({"error": "request too large"}, status_code=413)
    try:
        body = json.loads(raw)
    except ValueError:
        return JSONResponse({"error": "the body must be JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "the body must be a JSON object"}, status_code=400)
    text = body.get("text", "")
    if not isinstance(text, str) or len(text) > MAX_TEXT:
        return JSONResponse({"error": f"the text must be a string of at most {MAX_TEXT} characters"},
                            status_code=400)
    return body


@app.get("/")
async def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/samples")
async def samples():
    return SAMPLES


@app.get("/api/models")
async def models():
    return MODELS


@app.post("/api/parse")
async def parse_only(request: Request):
    body = await read_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        return parse(body.get("text", "")).to_dict()
    except ParseError as exc:
        return JSONResponse(exc.to_dict(), status_code=400)


@app.post("/api/ask")
async def ask(request: Request):
    from decisions import Choice
    body = await read_body(request)
    if isinstance(body, JSONResponse):
        return body
    model = str(body.get("model") or MODELS[0]["id"])
    if model not in {m["id"] for m in MODELS}:
        return JSONResponse({"error": f"unknown model {model!r}"}, status_code=400)
    images = body.get("images") or []
    if not isinstance(images, list) or len(images) > MAX_IMAGES:
        return JSONResponse({"error": f"at most {MAX_IMAGES} images"}, status_code=400)
    images = [u for u in images if isinstance(u, str) and u.startswith("data:image/")]
    if any(len(u) > MAX_IMAGE_URL for u in images):
        return JSONResponse({"error": f"an image is larger than {MAX_IMAGE_URL // 2**20} MB"},
                            status_code=400)
    try:
        query = parse(body.get("text", ""))
    except ParseError as exc:
        return JSONResponse(exc.to_dict(), status_code=400)
    questions = {q.key: Choice(instructions=q.text, criteria={o.name: o.description for o in q.options})
                 for q in query.questions}
    state = query.context or NO_CONTEXT
    if images and not next(m for m in MODELS if m["id"] == model)["vision"]:
        return JSONResponse({"error": f"{model} does not take images; pick a vision model or remove them"}, status_code=400)
    t0 = time.perf_counter()
    try:
        # The call blocks for a network round trip or more; in a thread, the
        # event loop keeps serving everyone else meanwhile.
        result = await asyncio.to_thread(engine(model).system_one, state, questions,
                                         images=tuple(images) or None)
    except Exception:
        log.exception("asking %s failed", model)
        return JSONResponse({"error": "the model could not be reached; try again"}, status_code=502)
    wall = time.perf_counter() - t0
    out = query.to_dict()
    for q in out["questions"]:
        answer = result.answers[q["key"]]
        q["choice"] = answer.choice
        q["confidence"] = round(answer.confidence, 3)
        q["probabilities"] = {name: round(p, 4) for name, p in answer.probabilities.items()}
    out.update({"model": model, "images": len(images), "ms": round(wall * 1000),
                "usage": {"input_tokens": result.usage.input_tokens, "output_tokens": result.usage.output_tokens},
                "state": state})
    return out
