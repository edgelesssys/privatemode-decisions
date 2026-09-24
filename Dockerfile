# Privatemode Decisions: the FastAPI backend and the decisions library.
FROM python:3.14-slim
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY pyproject.toml LICENSE ./
COPY decisions decisions
RUN pip install --no-cache-dir ".[demo]"
COPY app app
COPY web web
EXPOSE 8600
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8600"]
