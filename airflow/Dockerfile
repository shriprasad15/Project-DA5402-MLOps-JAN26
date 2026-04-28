FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV UV_HTTP_TIMEOUT=300

USER airflow
# PyPI packages
RUN --mount=type=cache,uid=50000,gid=0,target=/home/airflow/.cache/uv \
    uv pip install --python /home/airflow/.local/bin/python \
    loguru==0.7.2 pandas==2.2.3 pyarrow==17.0.0 numpy==2.1.3 \
    requests==2.32.3 scikit-learn==1.5.2 mlflow==3.11.1 \
    pydantic==2.9.2 pydantic-settings==2.5.2 python-dotenv==1.0.1 \
    transformers==4.44.2 tokenizers==0.19.1
# CPU-only torch (separate step to keep PyPI index clean)
RUN --mount=type=cache,uid=50000,gid=0,target=/home/airflow/.cache/uv \
    uv pip install --python /home/airflow/.local/bin/python \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.4.1+cpu"
