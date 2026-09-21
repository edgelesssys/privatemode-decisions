"""Turn an image into the data URL the chat API takes.

Kept separate from :mod:`system_one.inference` because it is the only part
of the package that has an optional dependency: Pillow, and only when a
resize is asked for. A caller who already has a JPEG of the right size
needs nothing installed.

The default is to send what you have. An image becomes vision tokens
roughly in proportion to its area and those tokens are prefilled like any
other, but against a ~200 ms round trip that prefill is invisible, so
downscaling is an optimisation rather than the normal path. ``max_side``
is there for when the tokens themselves are the cost -- a high decision
rate, a metered endpoint, or a server close enough that prefill stops
hiding. ``bench/vision.py`` measures the curve.
"""

from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path
from typing import Any

def to_data_url(source: Any, *, max_side: int | None = None, quality: int = 80) -> str:
    """Return ``data:image/...;base64,...`` for ``source``.

    ``source`` may be a data URL or ``http(s)`` URL (returned unchanged, so
    a caller who already has one pays nothing), a path, raw bytes, or a
    Pillow image. ``max_side`` scales the longest edge down to that many
    pixels -- never up, since inventing pixels only costs tokens.
    """
    if isinstance(source, str) and source.startswith(("data:", "http://", "https://")):
        return source

    if hasattr(source, "save") and hasattr(source, "size"):  # a Pillow image
        return _encode(source, max_side, quality)

    if isinstance(source, (bytes, bytearray)):
        raw, guessed = bytes(source), "image/jpeg"
    else:
        path = Path(source)
        raw = path.read_bytes()
        guessed = mimetypes.guess_type(path.name)[0] or "image/jpeg"

    if max_side is None:
        return f"data:{guessed};base64," + base64.b64encode(raw).decode()
    return _encode(_open(raw), max_side, quality)


def _open(raw: bytes):
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("Resizing an image needs Pillow: pip install Pillow") from exc
    return Image.open(io.BytesIO(raw))


def _encode(image: Any, max_side: int | None, quality: int) -> str:
    image = image.convert("RGB")
    if max_side:
        width, height = image.size
        longest = max(width, height)
        if longest > max_side:
            scale = max_side / longest
            image = image.resize((max(1, round(width * scale)),
                                  max(1, round(height * scale))))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
