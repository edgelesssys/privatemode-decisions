"""The demo backend: parse the text box, ask s1, return the distributions.

    SYSTEM_ONE_BASE_URL=http://proxy:8082/v1 SYSTEM_ONE_API_KEY=... uvicorn app.main:app

Both variables are also read from a ``.env`` file in the repository root.
The library is ``system_one/`` in this repository: the s1 client, taken as
it is.
"""

from __future__ import annotations

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
NO_CONTEXT = "Answer from your own judgment; there is no further context."


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


load_env()

app = FastAPI(title="s1 demo")
_engines: dict[str, object] = {}
_lock = threading.Lock()


def engine(model: str):
    """One SystemOne per model, so the token oracle's cache is reused."""
    from system_one import SystemOne
    with _lock:
        if model not in _engines:
            _engines[model] = SystemOne.from_env(model)
        return _engines[model]


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
    body = await request.json()
    try:
        return parse(str(body.get("text", ""))).to_dict()
    except ParseError as exc:
        return JSONResponse(exc.to_dict(), status_code=400)


@app.post("/api/ask")
async def ask(request: Request):
    from system_one import Choice
    body = await request.json()
    model = str(body.get("model") or MODELS[0]["id"])
    if model not in {m["id"] for m in MODELS}:
        return JSONResponse({"error": f"unknown model {model!r}"}, status_code=400)
    images = [u for u in body.get("images") or [] if isinstance(u, str) and u.startswith("data:image/")][:MAX_IMAGES]
    try:
        query = parse(str(body.get("text", "")))
    except ParseError as exc:
        return JSONResponse(exc.to_dict(), status_code=400)
    questions = {q.key: Choice(instructions=q.text, criteria={o.name: o.description for o in q.options})
                 for q in query.questions}
    state = query.context or NO_CONTEXT
    if images and not next(m for m in MODELS if m["id"] == model)["vision"]:
        return JSONResponse({"error": f"{model} does not take images; pick a vision model or remove them"}, status_code=400)
    t0 = time.perf_counter()
    try:
        result = engine(model).system_one(state, questions, images=tuple(images) or None)
    except Exception as exc:  # noqa: BLE001 - the proxy's words are the useful part
        return JSONResponse({"error": f"{type(exc).__name__}: {str(exc)[:400]}"}, status_code=502)
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
