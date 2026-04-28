# Architecture — Passive-Aggressive Email Detector (DA5402 Wave 3)

## 1. System Overview

The PA Detector is an **on-premises MLOps system** deployed entirely via Docker
Compose. No cloud provider is used: all compute, storage, networking, and model
artefacts reside on a single machine. The stack exposes eight user-facing ports and
one internal Postgres port, and coordinates training, serving, monitoring, and user
interaction across **nine cooperating services**.

The system satisfies the full DA5402 rubric surface area:

| Rubric area | Implementation |
|---|---|
| Data pipeline | DVC 6-stage pipeline (`dvc.yaml`) |
| Model training | Multi-task DistilBERT, MLflow run tracking |
| Model serving | `mlflow models serve` + FastAPI orchestration layer |
| Frontend | Streamlit single-page app |
| Monitoring | Prometheus + Grafana + Alertmanager |
| Orchestration | Apache Airflow (LocalExecutor, daily schedule) |
| Data versioning | DVC with SHA-256 content hashes |
| Feedback loop | PostgreSQL `predictions.user_feedback` column |

---

## 2. Architecture Diagram

```
                         ┌─────────────────────┐
                         │    User Browser      │
                         └──────────┬──────────┘
                                    │  HTTP :8501
                         ┌──────────▼──────────┐
                         │  Streamlit Frontend  │
                         │      :8501           │
                         │  (frontend/app.py)   │
                         └──────────┬──────────┘
                                    │  REST /predict  /feedback
                                    │  ($BACKEND_URL env var)
                         ┌──────────▼──────────┐
                         │  FastAPI Backend     │
                         │      :8000           │
                         │  (backend/app/)      │
                         └───┬───────┬──────────┘
                             │       │
             POST /invocations│       │ SQLAlchemy ORM
                             │       │
              ┌──────────────▼──┐  ┌─▼──────────────────────┐
              │  MLflow Model   │  │  PostgreSQL :5432        │
              │  Server :8080   │  │  ┌──────────────────┐   │
              │  (pa-detector/  │  │  │ predictions table │   │
              │   Production)   │  │  ├──────────────────┤   │
              └──────────┬──────┘  │  │ mlflow metadata  │   │
                         │         │  ├──────────────────┤   │
              ┌──────────▼──────┐  │  │ airflow metadata │   │
              │  MLflow Tracking│  └──┴──────────────────┴───┘
              │  Server :5000   │
              │  (pa-detector   │
              │   experiment)   │
              └─────────────────┘

┌─────────────────────┐   BashOperator shell-out   ┌────────────────────────┐
│  Airflow Webserver  │ ─────── triggers ─────────►│  DVC Pipeline          │
│      :8080          │                             │  ingest                │
│  DAG: pa_training_  │                             │    └─► clean           │
│  pipeline (@daily)  │                             │          └─► label_map │
└─────────────────────┘                             │                └─► drift_baseline
                                                    │                      └─► train
                                                    │                            └─► evaluate
                                                    └──────────┬─────────────────────────┘
                                                               │  mlflow run . -e train
                                                    ┌──────────▼──────────┐
                                                    │  MLflow Experiment   │
                                                    │  "pa-detector"       │
                                                    │  params / metrics /  │
                                                    │  artifacts / tags    │
                                                    └─────────────────────┘

┌─────────────────────┐   scrape /metrics every 15s
│  Prometheus :9090   │◄────────────────────────────  FastAPI :8000/metrics
│  (2 alert rules)    │
└──────────┬──────────┘
           │  PromQL queries
┌──────────▼──────────┐
│  Grafana :3000       │   API Health + ML Health dashboards (10s refresh)
└─────────────────────┘

┌─────────────────────┐
│  Alertmanager :9093  │◄──── firing alerts ──── Prometheus
└──────────┬──────────┘
           │  POST webhook
┌──────────▼──────────┐
│  FastAPI /admin/alert│  (logs alert body via loguru)
└─────────────────────┘
```

> **Diagram artefacts:** `docs/img/` is the intended location for supplementary
> screenshots. Run `dvc dag --dot | dot -Tpng -o docs/img/dvc_dag.png` after any
> pipeline change to regenerate the DVC DAG image. An Airflow screenshot can be
> captured from `http://localhost:8081`.

---

## 3. Component Descriptions

### 3.1 postgres:16

**Image:** `postgres:16-alpine` | **Internal port:** 5432

Central data store for three distinct schemas sharing one instance:

| Schema | Owned by | Purpose |
|---|---|---|
| `mlops` (default DB) | MLflow + Backend | MLflow experiment/run metadata; `predictions` table |
| `airflow` | Airflow | DAG state, task instance logs, XCom store |

A named Docker volume (`pg_data`) persists data across container restarts.
Health-checked via `pg_isready -U mlops` before any dependent service starts.
All other services declare `depends_on: postgres: condition: service_healthy`.

### 3.2 mlflow (Tracking Server + Model Registry)

**Image:** `ghcr.io/mlflow/mlflow:v2.17.0` | **Port:** 5000

Runs `mlflow server` with `--backend-store-uri postgresql+psycopg2://...` pointing
at Postgres and `--default-artifact-root /mlruns` (a named Docker volume). Provides:

- **Experiment tracking** — parameters, metrics, tags logged by `src/train.py`
- **Model Registry** — `pa-detector` registered model with `Production` stage alias
- **Artifact store** — PyTorch model checkpoints and `eval.json` files

### 3.3 model-server

**Image:** `ghcr.io/mlflow/mlflow:v2.17.0` | **Port:** 8080

Runs `mlflow models serve -m models:/pa-detector/Production -p 8080 --env-manager local`.
Exposes a `/invocations` REST endpoint that the FastAPI backend calls when
`MODEL_CLIENT=http`. In the default Docker Compose configuration `MODEL_CLIENT=mock`
is set, so this service starts but the backend uses `MockModelClient` instead,
allowing the full UI/API stack to operate without a trained model.

### 3.4 backend (FastAPI)

**Image:** Custom build from `backend/Dockerfile` | **Port:** 8000

The orchestration hub of the serving path. Responsibilities:

- Validates all request/response contracts via Pydantic v2 schemas (`backend/app/schemas.py`)
- Calls the Model Server (HTTP) or `MockModelClient` (development) and persists results
- Exposes **eight Prometheus metrics** via `/metrics` (see §3.7 and HLD §6)
- Runs the rolling-window `DriftMonitor` on every `/predict` request
- Serves the Alertmanager webhook receiver at `POST /admin/alert`
- Provides liveness (`/health`) and readiness (`/ready`) probes for Docker health checks
- Structured JSON logging via `loguru`

CORS is fully open (`allow_origins=["*"]`) to permit cross-origin calls from Streamlit.

### 3.5 frontend (Streamlit)

**Image:** Custom build from `frontend/Dockerfile` | **Port:** 8501

A single-page Streamlit application (`frontend/app.py`):

- `st.text_area` accepts raw email body (up to 5000 characters)
- POSTs to `$BACKEND_URL/predict` with `X-Correlation-Id` header
- Renders PA score, sarcasm score, tone label with colour badge, colour-coded
  highlighted phrase spans, and "Honest Translation" block
- Thumbs-up / thumbs-down feedback buttons wire to `POST /feedback`
- Sidebar links to MLflow (:5000), Airflow (:8080), Grafana (:3000), Prometheus (:9090)
  satisfy the rubric's ML Pipeline Visualization criterion

The only coupling to the backend is through the `BACKEND_URL` environment variable;
Streamlit imports no backend Python code.

### 3.6 airflow-webserver

**Image:** Custom build from `airflow/Dockerfile` | **Port:** 8080

Apache Airflow 2.x with `AIRFLOW__CORE__EXECUTOR=LocalExecutor`. Tasks run as
sub-processes on the same machine — no Celery worker or Redis broker required. The
single DAG `pa_training_pipeline` (`airflow/dags/training_pipeline.py`):

- Scheduled `@daily`, `catchup=False`, `retries=1`
- Six `BashOperator` tasks shell out to `python -m src.data.*` and `mlflow run`
- Task order mirrors DVC stage dependencies exactly

Airflow metadata is stored in a dedicated `airflow` database within the shared
Postgres instance.

### 3.7 prometheus

**Image:** `prom/prometheus:v2.52.0` | **Port:** 9090

Pull-based metrics collection. Configuration in `monitoring/prometheus/prometheus.yml`:

- **Scrape interval:** 15 seconds
- **Scrape targets:** `backend:8000/metrics`, `mlflow:5000/metrics`, `localhost:9090`
- **Alert rules:** loaded from `monitoring/prometheus/alert_rules.yml`
  - `HighErrorRate` — HTTP 5xx rate > 5% for 2 minutes
  - `DriftDetected` — `drift_input_length_ks_pvalue < 0.01` for 5 minutes
- Sends firing alerts to `alertmanager:9093`

### 3.8 grafana

**Image:** `grafana/grafana:11.1.0` | **Port:** 3000

Two dashboards are provisioned automatically at container start from JSON files:

| Dashboard | File | Key panels |
|---|---|---|
| API Health | `monitoring/grafana/dashboards/api-health.json` | Request rate, p95 latency |
| ML Health | `monitoring/grafana/dashboards/ml-health.json` | Inference p95, drift KS p-value, OOV rate, feedback counts |

Both dashboards refresh every **10 seconds**. Grafana datasource provisioning
(`monitoring/grafana/provisioning/datasources/prometheus.yml`) points at
`http://prometheus:9090` at startup — no manual configuration required.

Default credentials: `admin` / `admin` (override with `GF_SECURITY_ADMIN_PASSWORD`).

### 3.9 alertmanager

**Image:** `prom/alertmanager:v0.27.0` | **Port:** 9093

Receives firing alerts from Prometheus and forwards them to the backend webhook
(`POST http://backend:8000/admin/alert`), which logs them with severity context
via `loguru`. Configuration in `monitoring/alertmanager/config.yml`. The
`/admin/alert` route is protected by an `X-Admin-Token` header check
(`settings.BACKEND_ADMIN_TOKEN`).

---

## 4. DVC Pipeline Stages

Defined in `dvc.yaml`. Visualise with `dvc dag`. Six stages execute sequentially:

```
ingest
  │
  ▼
clean
  │
  ▼
label_map
  │
  ▼
drift_baseline
  │
  ▼
train  ←── mlflow run . -e train
  │
  ▼
evaluate
```

| Stage | Command | Inputs | Outputs |
|---|---|---|---|
| `ingest` | `python -m src.data.ingest` | Source adapters (HuggingFace, GitHub URLs) | `data/raw/` |
| `clean` | `python -m src.data.clean` | `data/raw/` | `data/interim/` |
| `label_map` | `python -m src.data.label_map` | `data/interim/` | `data/processed/` (train / val / test parquet, 80/10/10 split) |
| `drift_baseline` | `python -m src.data.drift_baseline` | `data/processed/` | `data/reference/feature_stats.json` |
| `train` | `mlflow run . -e train` | `data/processed/`, `src/models/` | `models/checkpoint.pt`, `metrics.json` |
| `evaluate` | `python -m src.evaluate` | `models/`, `data/processed/` | `eval.json` |

An optional `synth_pa` stage (`src/data/synthesize.py`) and a `quantize` stage
are present in `dvc.yaml` but not wired to the Airflow DAG — see `docs/future-work.md`.

DVC tracks all `data/` and `models/` paths with SHA-256 content hashes stored in
`.dvc` sidecar files, providing Git-like reproducibility for large binaries without
bloating the repository.

> Run `dvc dag --dot | dot -Tpng -o docs/img/dvc_dag.png` to generate the pipeline
> diagram. The PNG should be committed at `docs/img/dvc_dag.png`.

---

## 5. Data Flow

```
[Source datasets]
  sarcasm_headlines (Misra 2019, GitHub JSON)
  GoEmotions (HuggingFace)
  iSarcasm (HuggingFace mirror)
  Enron subset (email corpus)
        │  ingest stage: src/data/ingest.py + src/data/sources.py
        ▼
[data/raw/]  ──────────────────  Parquet per source, cached between runs
        │  clean stage: src/data/clean.py
        │  URLs → <URL>, emails → <EMAIL>, whitespace normalised, length-filtered
        ▼
[data/interim/]
        │  label_map stage: src/data/label_map.py
        │  Maps source-native labels → unified schema:
        │  [text, passive_aggression∈[0,1], sarcasm∈[0,1], tone∈{0..4}, source, weak_label]
        ▼
[data/processed/train|val|test.parquet]   (80 / 10 / 10 split, seed=42)
        │  drift_baseline stage: src/data/drift_baseline.py
        ▼
[data/reference/feature_stats.json]  ─── length_mean, length_std, length_quantiles, vocab
        │  train stage: src/train.py via mlflow run
        │
        ├── DistilBERT tokenizer (distilbert-base-uncased, max_length=128 CLI / 256 params.yaml)
        │     UnifiedDataset → DataLoader (batch_size=64 params.yaml)
        │
        ▼
[PassiveAggressiveDetector.forward(input_ids, attention_mask)]
   DistilBertModel → last_hidden_state[:, 0, :] (CLS)
   → nn.Dropout(0.1)
   → pa_head     nn.Linear(768,1)  → sigmoid → pa_score ∈ [0,1]
   → sarcasm_head nn.Linear(768,1) → sigmoid → sarcasm_score ∈ [0,1]
   → tone_head   nn.Linear(768,5)  → argmax  → tone_class ∈ {0..4}

   UncertaintyWeightedLoss (Kendall & Gal 2018) — 3 learned log_sigma params
        │
        ▼
[models/checkpoint.pt]  ←── mlflow.pytorch.log_model → MLflow Registry (pa-detector)
        │  Inference path (runtime)
        ▼
[FastAPI POST /predict]
   model_client.predict(text) → ModelResponse
   SQLAlchemy INSERT → predictions table (PostgreSQL)
   DriftMonitor.update(text) → Prometheus Gauges
   attributions_to_highlighted_phrases() → highlighted span list
        │
        ▼
[PredictResponse JSON]
   {prediction_id, scores:{passive_aggression, sarcasm}, tone, tone_confidence,
    highlighted_phrases, translation, model_version, latency_ms}
        │
        ▼
[Streamlit Browser]
   PA score bar, sarcasm score bar, tone badge, highlighted email text, translation
   👍/👎 feedback → POST /feedback → predictions.user_feedback (PostgreSQL)
        │
        ▼
[Prometheus scrapes /metrics every 15s → Grafana dashboards → Alertmanager rules]
```

---

## 6. Technology Choices Rationale

### DistilBERT (`distilbert-base-uncased`)

DistilBERT is 40% smaller (66M vs 110M parameters) and 60% faster than BERT-base
while retaining 97% of GLUE benchmark performance (Sanh et al., 2019). For a
CPU-deployable on-prem system this is the correct trade-off between capability and
inference latency (~50 ms CPU p50 with MockModelClient; ~80–120 ms with real
DistilBERT on CPU). Its 768-dimensional CLS token is shared by three independent
linear heads, making multi-task inference a single forward pass with no ensemble
overhead.

### Multi-task Learning with UncertaintyWeightedLoss

Jointly optimising PA regression, sarcasm regression, and 5-class tone
classification allows the model to learn shared linguistic features (hedging cues,
irony markers, softening language). The uncertainty-weighted loss (Kendall & Gal,
NeurIPS 2018) learns per-task log-variance parameters (`log_sigma[0..2]`)
automatically, removing the need to hand-tune loss weight hyperparameters across the
three heterogeneous tasks.

### Streamlit Frontend

Streamlit produces an interactive single-page UI with approximately 120 lines of
Python and requires no JavaScript build step. The only coupling point to the backend
is the `BACKEND_URL` environment variable — Streamlit imports no backend code. This
satisfies the rubric's frontend requirement while preserving loose coupling.

### LocalExecutor Airflow

The on-prem single-machine constraint rules out Celery + Redis or
Kubernetes-based executors. LocalExecutor runs task sub-processes on the host OS
without additional infrastructure. The six-task DAG order (`ingest >> clean >>
label_map >> drift_baseline >> train >> evaluate`) mirrors DVC stage dependencies
exactly, providing a natural migration path to CeleryExecutor when horizontal
scaling is required.

### DVC for Data Versioning

DVC provides Git-like semantics for large files (Parquet datasets, model
checkpoints) without bloating the Git repository. `dvc repro` re-executes only
stale stages — a stage is considered stale if its input hash or command has changed.
The `.dvc` sidecar files are committed to Git alongside code, binding a specific
data version to every code commit. `dvc push` / `dvc pull` enable sharing data
artefacts across machines via a configured remote.

### Prometheus + Grafana

The Prometheus pull model is the industry standard for container-native metrics.
The `prometheus_client` Python library integrates into FastAPI with a single
`/metrics` endpoint — no sidecar agent required. Grafana's provisioning-as-code
approach (JSON dashboards + YAML datasource config) means dashboards are
version-controlled and auto-loaded at container start with no manual UI setup.

### Feedback Loop to PostgreSQL

Storing user thumbs-up / thumbs-down votes in the `predictions.user_feedback`
column creates a labelled dataset of real-world predictions. This is the foundation
for future automated retraining triggers: if the down-vote rate exceeds a threshold,
an Airflow DAG run can be triggered programmatically.

---

## 7. Deployment

```bash
# First-run setup (see SETUP.md)
conda env create -f conda.yaml
conda activate mlops

# Start all 9 services and wait for health checks
docker compose up -d --wait --wait-timeout 180

# Verify all services are healthy
docker compose ps

# Trigger a training pipeline run manually
docker compose exec airflow-webserver \
    airflow dags trigger pa_training_pipeline

# Regenerate DVC pipeline diagram
dvc dag --dot | dot -Tpng -o docs/img/dvc_dag.png

# View live metrics
# Prometheus:   http://localhost:9090
# Grafana:      http://localhost:3000   (admin / admin)
# MLflow:       http://localhost:5000
# Airflow:      http://localhost:8081   (admin / admin)
# Frontend:     http://localhost:8501
# Backend API:  http://localhost:8000/docs
```

See `SETUP.md` for full first-run instructions including conda environment setup,
`.env` configuration, and DVC remote initialisation.
