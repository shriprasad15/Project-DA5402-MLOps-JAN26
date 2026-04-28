FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV UV_HTTP_TIMEOUT=300

WORKDIR /app
RUN groupadd -r appuser && useradd -r -g appuser appuser

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    loguru==0.7.2 \
    numpy==2.1.3 \
    pandas==2.2.3 \
    prometheus-client==0.21.0 \
    pyarrow==17.0.0 \
    pydantic==2.9.2 \
    pydantic-settings==2.5.2 \
    python-dotenv==1.0.1 \
    requests==2.32.3 \
    "scipy>=1.13.0" \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.30.6" \
    sqlalchemy==2.0.35 \
    psycopg2-binary==2.9.9 \
    alembic==1.13.3 \
    httpx==0.28.1

COPY backend/ ./backend/
COPY contracts/ ./contracts/

USER appuser
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
