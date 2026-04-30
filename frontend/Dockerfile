FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV UV_HTTP_TIMEOUT=300

WORKDIR /app
ENV PYTHONPATH=/app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    streamlit==1.38.0 \
    requests==2.32.3 \
    pyyaml==6.0.2

COPY frontend/ ./frontend/
COPY monitoring/config.yaml ./monitoring/config.yaml

EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true"]
