FROM ghcr.io/mlflow/mlflow:v3.11.1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV UV_HTTP_TIMEOUT=300

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system psycopg2-binary==2.9.9

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    torch==2.4.1+cpu \
    transformers==4.44.2 \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY src/ /app/src/
ENV PYTHONPATH=/app
