# Setup

## Prerequisites

- macOS or Linux host
- Docker Desktop ≥ 4.30 (or Docker Engine ≥ 26) with Compose v2
- Python 3.11 (`pyenv install 3.11.9` recommended)
- Git + Git LFS (`brew install git-lfs && git lfs install`)
- (Optional but recommended) NVIDIA GPU driver ≥ 555 for CUDA 12.1 PyTorch wheels
- Ollama ≥ 0.3.6 with the `gemma3:4b` model pulled:
  ```bash
  ollama pull gemma3:4b
  ```

## Local bootstrap

```bash
git clone <repo>
cd passive-aggressive-email-detector
cp .env.example .env
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
docker compose up -d
```

## Verify

```bash
curl -k https://localhost/api/health         # {"status":"ok"}
open http://localhost:5000                    # MLflow
open http://localhost:8080                    # Airflow
open http://localhost:3000                    # Grafana
```

## Troubleshooting

- **Compose boots slowly on first run** — images are downloading (~3 GB).
- **Airflow webserver restarts** — ensure `AIRFLOW__CORE__FERNET_KEY` is set
  (`python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`).
- **GPU not detected inside training container** — ensure NVIDIA Container
  Toolkit is installed; `docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi` must work.
