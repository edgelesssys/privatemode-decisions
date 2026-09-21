#!/bin/sh
# Start the demo on http://127.0.0.1:8600 (reads .env for the proxy URL and key).
cd "$(dirname "$0")"
exec .venv/bin/uvicorn app.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8600}"
