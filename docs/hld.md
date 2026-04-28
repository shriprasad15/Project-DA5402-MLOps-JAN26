# High-Level Design — Passive-Aggressive Email Detector (DA5402 Wave 3)

## 1. Problem Statement

Email communication frequently carries sub-textual cues — passive aggression,
sarcasm, and tonal ambiguity — that are invisible to rule-based keyword filters and
difficult even for human readers to identify reliably. Misread tone in professional
email leads to interpersonal friction and miscommunication.

The PA Detector addresses this problem by providing a **multi-task NLP classifier**
that analyses an email body (and optional subject line) and returns:

- A **passive-aggression score** in [0, 1] — quantifying how much the message
  relies on indirect, deniable hostility rather than direct assertion
- A **sarcasm score** in [0, 1] — quantifying how much the stated meaning diverges
  from the intended meaning
- A **tone classification** across five classes: Neutral, Friendly, Assertive,
  Aggressive, Passive-Aggressive
- **Highlighted phrases** identifying which tokens contributed most to the scores
  (rule-based span attribution in Wave 3; SHAP-based in future work)
- An **honest translation** rephrasing the implicit subtext as direct language

The system is designed to run entirely on-premises (no cloud dependency), to be
deployable with a single `docker compose up` command, and to satisfy the DA5402
MLOps rubric covering data pipelines, model training, serving, monitoring,
observability, and the full ML lifecycle.

---

## 2. Design Paradigm

The system uses a **hybrid paradigm**:

### Object-Oriented Service Layer

FastAPI router classes, SQLAlchemy ORM models (`backend/app/db/models.py`), Pydantic
v2 schema classes (`backend/app/schemas.py`), and dataclass-based service objects
(`DriftMonitor`, `MockModelClient`, `HTTPModelClient`) follow OO principles for
encapsulation and testability. The `contracts/` package provides a shared `Tone` enum
and JSON schema imported by both backend and training code — a single source of truth
that prevents schema drift across service boundaries.

### Functional Pipeline Modules

All DVC pipeline stages (`src/data/ingest.py`, `src/data/clean.py`,
`src/data/label_map.py`, `src/data/drift_baseline.py`, `src/train.py`,
`src/evaluate.py`) are implemented as pure or near-pure functions with a `main()`
entry point. Each stage is independently testable, rerunnable without side effects on
other stages, and composable via DVC's dependency graph.

---

## 3. Key Design Choices

| Choice | Rationale |
|---|---|
| DistilBERT multi-task | Lightweight transformer (66M params), 3 simultaneous outputs from one CLS token forward pass, pretrained on English general corpus; 60% faster than BERT-base at 97% capability (Sanh et al., 2019) |
| Uncertainty-weighted loss | Automatically balances 3 task gradients via 3 learned log-variance parameters — removes manual loss weight tuning and adapts to task difficulty during training (Kendall & Gal, NeurIPS 2018) |
| Streamlit frontend | Single-page app, fast to develop, rubric-compliant, no JS build step; loose coupling via `BACKEND_URL` env var — Streamlit imports no backend code |
| LocalExecutor Airflow | On-prem single-machine constraint; no Celery/Redis overhead; 6-task DAG order mirrors DVC stage dependencies exactly |
| DVC for data versioning | Git-like semantics for Parquet files and model checkpoints; reproducible pipeline with `dvc repro`; `.dvc` sidecar files bind a data version to every code commit |
| Prometheus + Grafana | Industry-standard pull-based monitoring; integrates with FastAPI `/metrics` endpoint via `prometheus_client`; dashboards provisioned as code (no manual UI setup) |
| Feedback loop to PostgreSQL | Captures user thumbs-up/down votes in `predictions.user_feedback`; foundation for future automated retraining trigger |
| PostgreSQL as single data store | One `postgres:16` instance serves MLflow metadata, Airflow task state, and Backend predictions — minimises operational complexity on single-machine deployment |
| `contracts/` shared package | `Tone` enum and JSON schema (`contracts/schemas.json`) imported by both backend and training code — eliminates schema drift across service boundaries |
| `MockModelClient` default | `MODEL_CLIENT=mock` in compose lets the full UI/API stack run without a trained model, enabling fast integration testing and frontend development |

---

## 4. Pipeline Performance Estimates

| Stage | Estimate | Notes |
|---|---|---|
| Ingest (all 4 adapters) | 5–10 min first run | Network download from HuggingFace + GitHub; subsequent runs are cached |
| Clean + label_map | ~2,000 rows/min | Pandas `map` on CPU; regex patterns compiled once at module level |
| Drift baseline | < 10 s | Single-pass quantile computation over processed parquet |
| Training (1 epoch, CPU, batch 16) | ~20 min per 10k rows | DistilBERT CPU-only; GPU with bf16 reduces to ~3 min |
| Training (5 epochs, GPU bf16, batch 64) | ~15 min per 10k rows | `params.yaml` default configuration |
| Inference latency CPU (MockModelClient) | ~2–5 ms p50 | Rule-based mock; validates full API + DB path |
| Inference latency CPU (real DistilBERT) | ~80–120 ms p50 | 66M-param forward pass without CUDA |
| Inference latency GPU bf16 | ~8–12 ms p50 | RTX-class GPU, single-sample inference |
| Streamlit end-to-end round trip | ~200 ms | Includes network + Pydantic validation + DB write + Prometheus update |

---

## 5. Loose Coupling Strategy

Services communicate exclusively through well-defined HTTP APIs or environment
variables. There is no shared memory or shared Python imports across service
boundaries at runtime:

| From | To | Coupling mechanism |
|---|---|---|
| Streamlit | Backend | `POST $BACKEND_URL/predict` — env var configures URL |
| Backend | Model Server | `POST $MODEL_SERVER_URL/invocations` — env var; falls back to `MockModelClient` |
| Backend | PostgreSQL | SQLAlchemy connection string assembled from `POSTGRES_*` env vars |
| Prometheus | Backend | Pull scrape of `GET :8000/metrics` — backend is unaware of Prometheus |
| Alertmanager | Backend | `POST /admin/alert` webhook — backend logs but does not act |
| Airflow | Pipeline | `BashOperator` tasks shell out to `python -m ...` — no Python imports |
| MLflow Model Server | MLflow Tracking | Reads artefacts from shared `mlflow_artifacts` Docker volume |

---

## 6. Monitoring Strategy

Eight Prometheus metrics are instrumented in `backend/app/observability/metrics.py`:

| Metric | Type | Labels | Purpose |
|---|---|---|---|
| `http_requests_total` | Counter | `service`, `endpoint`, `status` | Request rate and HTTP error rate (5xx ratio) |
| `http_request_duration_seconds` | Histogram | `service`, `endpoint` | Latency percentiles (p50, p95, p99) |
| `model_inference_duration_seconds` | Histogram | — | Model server or mock client round-trip time |
| `feedback_votes_total` | Counter | `vote` (`up`/`down`) | User satisfaction signal over time |
| `drift_input_length_ks_pvalue` | Gauge | — | Data drift — KS test p-value on rolling text lengths vs reference |
| `drift_oov_rate` | Gauge | — | Vocabulary drift — out-of-vocabulary token fraction (Wave 3 placeholder: 0.0) |
| `drift_pred_class` | Gauge | `tone` | Prediction class distribution shift |
| `drift_confidence_mean` | Gauge | — | Mean tone confidence — degradation signals model uncertainty |

Two alert rules in `monitoring/prometheus/alert_rules.yml`:

1. **HighErrorRate** — `sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.05` sustained for **2 minutes** → `severity: warning`
2. **DriftDetected** — `drift_input_length_ks_pvalue < 0.01` sustained for **5 minutes** → `severity: warning`

Grafana provides near-real-time visibility with a **10-second** dashboard refresh on
both the **API Health** dashboard (request rate, p95 latency) and the **ML Health**
dashboard (inference duration p95, drift KS p-value, OOV rate, feedback vote counts).

---

## 7. Problems Encountered and Mitigations

### 7.1 Sarcasm Dataset v2 URL Not Found

The iSarcasm v2 dataset URL returned HTTP 404 during Wave 1 data ingestion.

**Mitigation:** The `ISarcasmAdapter` in `src/data/sources.py` was patched to fall
back to the `jkhedri/psychology-dataset` HuggingFace mirror with a synthetic
`sarcasm=0.8` label, and a code comment marks the placeholder. The
`SarcasmHeadlinesAdapter` (Misra 2019, GitHub raw URL) was unaffected and remains
the primary sarcasm source.

### 7.2 Python 3.14 on System Python

The system Python was 3.14, which is incompatible with several compiled dependencies
(`torch`, `transformers`). `pyproject.toml` declares `requires-python = ">=3.11,<3.12"`.

**Mitigation:** A conda environment pinned to Python 3.11.15 was created via
`conda env create -f conda.yaml`. All subsequent development and CI runs use
`conda activate mlops` before any Python invocation. The `.github/workflows/ci.yml`
explicitly sets `python-version: "3.11"`.

### 7.3 No Docker Compose Plugin (Legacy `docker-compose` Binary)

The host machine had the legacy `docker-compose` v1 binary but not the modern
`docker compose` plugin (v2).

**Mitigation:** The `docker-compose.yml` was validated offline via
`python -c "import yaml; yaml.safe_load(open('docker-compose.yml'))"` and `SETUP.md`
was updated to note that either binary is acceptable. CI workflows use
`docker/setup-buildx-action` which provides the v2 plugin. No Compose-version-specific
syntax was used in the YAML.

---

## 8. Scalability Notes

The current deployment is deliberately minimal for an on-prem single-machine context.
Each component has a clear, low-friction upgrade path:

| Current | Upgrade path | Trigger condition |
|---|---|---|
| LocalExecutor Airflow | CeleryExecutor + Redis broker | > 1 concurrent DAG run or multi-machine deployment |
| Single Docker Compose | Docker Swarm or Kubernetes | > 1 host, HA requirement |
| SQLite fallback (unit tests) | Already on Postgres in compose | Already done for production path |
| `MockModelClient` | `HTTPModelClient` pointing at `model-server:8080` | Trained model promoted to `Production` in MLflow registry |
| Rule-based phrase highlighter | SHAP token attributions | Post-Wave-3 explainability work — see `docs/future-work.md` |
| Single Postgres instance | Read replica + PgBouncer | > 500 sustained req/s |
| No auth on backend | JWT authentication on admin routes | Multi-tenant deployment |
| No caching | Redis prediction cache for repeated inputs | High-volume repeated queries |
