# ML Pipeline Visualization Guide (DA5402 Wave 3)

This page explains how to navigate the training and serving pipeline across the
four tooling UIs. All URLs assume services are running via `docker compose up -d --wait`.

---

## Pipeline Overview

```
[Raw Data Sources]
  sarcasm_headlines (Misra 2019, GitHub)
  GoEmotions (HuggingFace)
  iSarcasm (HuggingFace mirror)
  Enron email subset
          |
          v
 ┌─────────────────────────────────────────────────────┐
 │              DVC Pipeline  (dvc.yaml)                │
 │                                                      │
 │  [ingest]  ──► data/raw/                            │
 │     |                                                │
 │  [clean]   ──► data/interim/                        │
 │     |                                                │
 │  [label_map] ──► data/processed/train|val|test.parquet│
 │     |                                                │
 │  [drift_baseline] ──► data/reference/feature_stats.json│
 │     |                                                │
 │  [train]  ──► models/checkpoint.pt                  │
 │     └─── mlflow run . -e train ──────────────────►  │
 │                                                  │   │
 │  [evaluate] ──► eval.json                        │   │
 └──────────────────────────────────────────────────│───┘
                                                    │
                                            ┌───────▼────────┐
                                            │ MLflow Tracking │
                                            │   :5000         │
                                            │ Experiment:     │
                                            │  "pa-detector"  │
                                            │ params/metrics/ │
                                            │ tags/artifacts  │
                                            └───────┬─────────┘
                                                    │ Model Registry
                                            ┌───────▼─────────┐
                                            │ pa-detector/    │
                                            │ Production      │
                                            └───────┬─────────┘
                                                    │
                                            ┌───────▼─────────┐
                                            │ model-server    │
                                            │    :8080        │
                                            │ mlflow models   │
                                            │ serve           │
                                            └───────┬─────────┘
                                                    │ POST /invocations
                                      ┌─────────────▼──────────┐
    User Request ────────────────────►│  FastAPI Backend :8000  │
                                      └──────┬────────┬─────────┘
                                             │        │
                                      ┌──────▼──┐  ┌──▼──────────┐
                                      │PostgreSQL│  │ /metrics    │
                                      │:5432    │  │ endpoint    │
                                      └─────────┘  └──────┬──────┘
                                                          │ scrape 15s
                                                   ┌──────▼──────┐
                                                   │ Prometheus  │
                                                   │   :9090     │
                                                   └──┬──────┬───┘
                                                      │      │ alert rules
                                               ┌──────▼──┐ ┌─▼──────────┐
                                               │ Grafana  │ │Alertmanager│
                                               │  :3000   │ │   :9093    │
                                               └──────────┘ └────────────┘
```

---

## 1. Airflow — Pipeline Orchestration

**URL:** http://localhost:8081  
**Login:** admin / admin

The `pa_training_pipeline` DAG runs `@daily` with `catchup=False`. Six tasks
execute sequentially:

| Task | Command | Input | Output |
|---|---|---|---|
| `ingest` | `python -m src.data.ingest` | Source adapters | `data/raw/` |
| `clean` | `python -m src.data.clean` | `data/raw/` | `data/interim/` |
| `label_map` | `python -m src.data.label_map` | `data/interim/` | `data/processed/` |
| `drift_baseline` | `python -m src.data.drift_baseline` | `data/processed/` | `data/reference/` |
| `train` | `mlflow run . -e train` | `data/processed/`, `src/models/` | `models/`, MLflow run |
| `evaluate` | `python -m src.evaluate` | `models/`, `data/processed/` | `eval.json` |

### How to Navigate

- **DAGs list** → click `pa_training_pipeline` → select **Graph** view to see the
  six-task chain and the directional dependencies
- **Click any task box** → select **Log** to view full stdout/stderr for that task
  instance
- **Calendar view** shows historical run outcomes: green = success, red = failed,
  grey = no run that day
- **Run history panel** tracks success/failure rate over time — this is the
  "console to track errors, failures, and successes" required by the rubric
- **Trigger manually:** click the play button (▶) at the top right of the DAG view,
  or run from the terminal:
  ```bash
  docker compose exec airflow-webserver \
      airflow dags trigger pa_training_pipeline
  ```

---

## 2. MLflow — Experiment Tracking and Model Registry

**URL:** http://localhost:5000

### Experiments Tab

- Select experiment **`pa-detector`** from the left panel
- Each row is one training run
- Default columns visible: `train_loss`, `val_macro_f1`, `pa_mae`, `sarcasm_mae`
- Additional tags logged per run: `git_sha`, `dataset_hash`, `hardware`
- Click a run name → **Artifacts** tab → download the saved PyTorch model directory
- Select two or more runs → click **Compare** to view metric diffs side-by-side
  and parameter tables

### Models Tab

- **`pa-detector`** registered model shows all versions
- The version tagged **Production** is what `model-server` currently serves
- To promote a new version: open the version → click **Stage: None** →
  **Transition to → Production**

### What is Logged (from `src/train.py`)

```python
mlflow.log_params(vars(args))              # all CLI arguments
mlflow.log_metric("train_loss", loss, step=epoch)
mlflow.log_metric("val_macro_f1", f1, step=epoch)
mlflow.log_metric("pa_mae", mae, step=epoch)
mlflow.log_metric("sarcasm_mae", mae, step=epoch)
mlflow.set_tag("git_sha", get_git_sha())
mlflow.set_tag("dataset_hash", get_file_hash(...))
mlflow.set_tag("hardware", args.device)
mlflow.pytorch.log_model(model, "model",
    registered_model_name="pa-detector")
mlflow.log_artifact("eval.json")
```

---

## 3. Grafana — Near Real-Time Monitoring

**URL:** http://localhost:3000  
**Login:** admin / admin

Two dashboards are pre-provisioned automatically from JSON under
`monitoring/grafana/dashboards/`. Both refresh every **10 seconds**.

### API Health Dashboard (`api-health.json`)

| Panel | PromQL expression | What to watch |
|---|---|---|
| Request Rate | `rate(http_requests_total[1m])` | Drop to 0 = service down |
| Request Duration p95 | `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[1m]))` | Should stay < 500 ms |
| Error Rate | `rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])` | Alert fires at > 5% |
| Feedback Votes | `increase(feedback_votes_total[1h])` | Thumbs-up/down ratio |

### ML Health Dashboard (`ml-health.json`)

| Panel | PromQL expression | What to watch |
|---|---|---|
| Inference Duration p95 | `histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket[1m]))` | GPU regression visible here |
| Drift KS p-value | `drift_input_length_ks_pvalue` | Alert fires below 0.01 |
| OOV Rate | `drift_oov_rate` | Vocabulary shift over time |
| Confidence Mean | `drift_confidence_mean` | Sustained drop = model uncertainty |
| Tone Distribution | `drift_pred_class` by label | Unexpected shift = distribution drift |

---

## 4. Prometheus — Raw Metrics and Alert Rules

**URL:** http://localhost:9090

### Useful Pages

- **Graph tab** — paste any metric name to plot it over time:
  - `http_requests_total`
  - `drift_input_length_ks_pvalue`
  - `model_inference_duration_seconds_bucket`
- **Alerts tab** — shows `HighErrorRate` and `DriftDetected` rules with their
  current state: `inactive` / `pending` / `firing`
- **Targets tab** — confirms `backend:8000/metrics` is being scraped successfully
  (should show `State: UP`)

### Alert Rules (from `monitoring/prometheus/alert_rules.yml`)

```yaml
HighErrorRate:
  expr: sum(rate(http_requests_total{status=~"5.."}[5m])) /
        sum(rate(http_requests_total[5m])) > 0.05
  for: 2m
  severity: warning
  fires to: alertmanager:9093 → POST backend:8000/admin/alert

DriftDetected:
  expr: drift_input_length_ks_pvalue < 0.01
  for: 5m
  severity: warning
  fires to: alertmanager:9093 → POST backend:8000/admin/alert
```

### All 8 Instrumented Metrics

| Metric name | Type | Labels |
|---|---|---|
| `http_requests_total` | Counter | `service`, `endpoint`, `status` |
| `http_request_duration_seconds` | Histogram | `service`, `endpoint` |
| `model_inference_duration_seconds` | Histogram | — |
| `feedback_votes_total` | Counter | `vote` |
| `drift_input_length_ks_pvalue` | Gauge | — |
| `drift_oov_rate` | Gauge | — |
| `drift_pred_class` | Gauge | `tone` |
| `drift_confidence_mean` | Gauge | — |

---

## 5. DVC — Data and Model Versioning

From the repository root (inside the conda `mlops` environment):

```bash
# Visualise the pipeline as an ASCII DAG
dvc dag

# Re-run only the stages whose inputs or commands have changed
dvc repro

# Check which stages are out-of-date without running them
dvc status

# Regenerate the pipeline diagram as a PNG (requires graphviz)
dvc dag --dot | dot -Tpng -o docs/img/dvc_dag.png
```

`dvc.lock` is committed after each successful `dvc repro` to pin the exact data and
model SHA-256 hashes — providing full reproducibility from any Git commit.

The DVC pipeline stages correspond 1:1 with the Airflow `pa_training_pipeline` tasks.
DVC manages file-level dependencies and caching; Airflow manages scheduling and
retries.
