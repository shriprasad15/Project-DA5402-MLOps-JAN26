FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV UV_HTTP_TIMEOUT=300

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    mlflow==3.11.1 \
    psycopg2-binary==2.9.9 \
    torch==2.4.1+cpu \
    transformers==4.44.2 \
    pandas==2.2.3 \
    numpy==2.1.3 \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY src/ /app/src/
ENV PYTHONPATH=/app
