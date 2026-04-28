# Low-Level Design — Passive-Aggressive Email Detector (DA5402 Wave 3)

## 1. API Endpoint Specifications

All endpoints are mounted on the FastAPI application defined in `backend/app/main.py`.

- **Base URL (Docker host):** `http://localhost:8000`
- **Base URL (container network):** `http://backend:8000`
- **OpenAPI spec:** `openapi.yaml` (root of repository)
- **Interactive docs:** `http://localhost:8000/docs` (Swagger UI)

| Endpoint | Method | Request Body | Response Body | Status Codes |
|---|---|---|---|---|
| `/health` | GET | — | `{"status": "ok"}` | 200 |
| `/ready` | GET | — | `{"status": "ready"}` | 200, 503 |
| `/predict` | POST | `PredictRequest` (see §2) | `PredictResponse` (see §3) | 200, 422 |
| `/feedback` | POST | `FeedbackRequest` (see §4) | — (empty body, 204 No Content) | 204, 404, 422 |
| `/metrics` | GET | — | Prometheus text/plain exposition format | 200 |
| `/admin/alert` | POST | Alertmanager webhook JSON body | `{"status": "received"}` | 200, 403 |

**Notes:**

- `/ready` returns 503 before the lifespan startup hook completes (DB engine + model
  client initialised). Returns 200 once `set_ready(True)` is called.
- `/admin/alert` is guarded by an `X-Admin-Token` header check against
  `settings.BACKEND_ADMIN_TOKEN` (default `"dev-token"`). Returns 403 if the header
  is absent or does not match.
- All 422 responses are Pydantic `RequestValidationError` bodies with field-level
  detail (location, message, type).
- Every request passes through the `metrics_middleware` which increments
  `http_requests_total` and records `http_request_duration_seconds`.

---

## 2. PredictRequest Schema

**Source:** `backend/app/schemas.py` → `class PredictRequest(BaseModel)`

| Field | Type | Constraints | Required |
|---|---|---|---|
| `text` | `str` | `min_length=1`, `max_length=5000`, must not be only whitespace (`not_only_whitespace` validator) | Yes |
| `subject` | `str \| None` | `max_length=500` | No (default `null`) |

The `@field_validator("text")` raises `ValueError("text must not be only whitespace")`
if `v.strip()` is empty; FastAPI converts this to a 422 response with field detail.

---

## 3. PredictResponse Schema

**Source:** `backend/app/schemas.py` → `class PredictResponse(BaseModel)`
**Also defined in:** `openapi.yaml` → `components/schemas/PredictResponse`

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `prediction_id` | `str` | — | UUID4 generated at request time (`str(uuid.uuid4())`) |
| `scores` | `dict[str, float]` | Keys: `passive_aggression`, `sarcasm`; values ∈ [0,1] | Populated directly from `ModelResponse` |
| `tone` | `Tone` (str Enum) | One of: `neutral`, `friendly`, `assertive`, `aggressive`, `passive_aggressive` | Defined in `contracts/tone_enum.py` |
| `tone_confidence` | `float` | `ge=0.0`, `le=1.0`; clamped to 1.0 in predict route | Softmax max probability for tone head |
| `highlighted_phrases` | `list[HighlightedPhrase]` | See sub-schema below | Token-attribution-derived character spans |
| `translation` | `str` | — | Honest-language rewrite from model or mock |
| `model_version` | `str` | — | Value stored in DB `model_version` column (default `"mock-v0"`) |
| `latency_ms` | `int` | `ge=0` | Wall-clock milliseconds for model client round-trip |

**HighlightedPhrase sub-schema** (`class HighlightedPhrase(BaseModel)`):

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `text` | `str` | — | The highlighted token or merged span text |
| `start` | `int` | `ge=0` | Start character offset in original request `text` |
| `end` | `int` | `ge=0` | End character offset (exclusive) in original request `text` |
| `severity` | `float` | `ge=0.0`, `le=1.0` | Attribution score; clamped to 1.0 in predict route via `min(p["severity"], 1.0)` |

---

## 4. FeedbackRequest Schema

**Source:** `backend/app/schemas.py` → `class FeedbackRequest(BaseModel)`

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `prediction_id` | `str` | `min_length=1` | Must match an existing row in `predictions` table |
| `vote` | `Literal["up", "down"]` | Only these two string values accepted | 422 returned for any other value |

**Feedback route behaviour** (`backend/app/api/feedback.py`):

1. `db.get(Prediction, body.prediction_id)` — lookup by primary key.
2. If `None` → HTTP 404 `{"detail": "prediction not found"}`.
3. `pred.user_feedback = body.vote` → `db.commit()`.
4. `feedback_votes_total.labels(vote=body.vote).inc()` — increments Prometheus counter.
5. Returns `Response(status_code=204)` — no body.

---

## 5. Database Schema

Single table `predictions` managed by SQLAlchemy ORM in `backend/app/db/models.py`.
Created via `Base.metadata.create_all(engine)` during the lifespan startup hook.
In Docker Compose the engine connects to PostgreSQL (`POSTGRES_HOST` env var set);
when `POSTGRES_HOST` is empty the backend falls back to a local SQLite file
(`pa_detector.db`) — used by unit tests via `TestClient`.

| Column | SQLAlchemy Type | Python Type | Nullable | Notes |
|---|---|---|---|---|
| `prediction_id` | `String` (PK) | `str` | No | UUID4, `default=lambda: str(uuid.uuid4())` |
| `created_at` | `DateTime` | `datetime` | No | `default=datetime.utcnow` |
| `correlation_id` | `String` | `str \| None` | Yes | From `X-Correlation-Id` request header |
| `text_hash` | `String` | `str` | No | SHA-256 hex of request `text`, truncated to 16 chars |
| `pa_score` | `Float` | `float` | No | Passive-aggression score ∈ [0,1] |
| `sarcasm_score` | `Float` | `float` | No | Sarcasm score ∈ [0,1] |
| `tone` | `String` | `str` | No | Tone enum string value |
| `tone_confidence` | `Float` | `float` | No | Tone softmax confidence ∈ [0,1] |
| `model_version` | `String` | `str` | No | Default `"mock-v0"` |
| `latency_ms` | `Integer` | `int` | No | Model client wall-clock latency in ms |
| `user_feedback` | `String` | `str \| None` | Yes | Null until `/feedback` called; set to `"up"` or `"down"` |

**Total: 11 columns.**

---

## 6. Model Architecture

**Source:** `src/models/multitask.py` → `class PassiveAggressiveDetector(nn.Module)`

```
Input tensors:
  input_ids      [B, L]  (long)
  attention_mask [B, L]  (long)
          │
          ▼
  DistilBertModel.from_pretrained("distilbert-base-uncased")
    66M parameters, 6 transformer layers, hidden_size=768
    → last_hidden_state: [B, L, 768]
          │
  CLS token slice: last_hidden_state[:, 0, :]
    → cls: [B, 768]
          │
  nn.Dropout(p=0.1)
          │
          ├──► pa_head:      nn.Linear(768, 1)
          │    inference:    torch.sigmoid(pa_logits).squeeze(-1) → pa_score ∈ [0,1]
          │    output shape: [B, 1]
          │
          ├──► sarcasm_head: nn.Linear(768, 1)
          │    inference:    torch.sigmoid(sarcasm_logits).squeeze(-1) → sarcasm_score ∈ [0,1]
          │    output shape: [B, 1]
          │
          └──► tone_head:    nn.Linear(768, 5)
               inference:    tone_logits.argmax(dim=-1) → tone_class ∈ {0,1,2,3,4}
               output shape: [B, 5]

Return: MultiTaskOutput(pa_logits, sarcasm_logits, tone_logits, hidden)
  hidden = cls (post-dropout CLS embedding)  [B, 768]
```

Constructor parameters:

| Parameter | Default | Notes |
|---|---|---|
| `pretrained` | `"distilbert-base-uncased"` | HuggingFace model ID |
| `num_tone_classes` | `5` | Must match `Tone` enum cardinality |
| `dropout` | `0.1` | Applied to CLS embedding before all three heads |
| `quantized` | `False` | Flag for future INT8 quantization support |

### 6.1 UncertaintyWeightedLoss

**Source:** `src/models/loss.py` → `class UncertaintyWeightedLoss(nn.Module)`

```
Parameters:
  log_sigma: nn.Parameter(torch.zeros(n_tasks))  — 3 learnable scalars

Forward formula:
  total = Σᵢ [ lossᵢ / (2 · exp(2 · log_sigma[i])) + log_sigma[i] ]

Where:
  loss[0] = binary_cross_entropy_with_logits(pa_logits,     pa_label)
  loss[1] = binary_cross_entropy_with_logits(sarcasm_logits, sarcasm_label)
  loss[2] = cross_entropy(tone_logits, tone_label)
```

The three `log_sigma` parameters are learned alongside model weights. This
implements the homoscedastic uncertainty weighting from Kendall & Gal (NeurIPS 2018),
which automatically calibrates the relative contribution of each task loss.

---

## 7. MLflow Experiment Structure

**Experiment name:** `pa-detector`
**Tracking URI:** `http://mlflow:5000` (compose) or `http://localhost:5000` (local)
**Source:** `src/train.py` — all tracking inside `with mlflow.start_run(run_name=...):`

### Logged Parameters (all `argparse` args)

| Parameter | CLI flag | Default |
|---|---|---|
| `data_path` | `--data-path` | `data/processed/train.parquet` |
| `val_path` | `--val-path` | `data/processed/val.parquet` |
| `epochs` | `--epochs` | `1` (Airflow DAG) / `5` (`params.yaml`) |
| `lr` | `--lr` | `2e-5` |
| `batch_size` | `--batch-size` | `16` (CLI) / `64` (`params.yaml`) |
| `device` | `--device` | `"cpu"` |
| `bf16` | `--bf16` | `False` |
| `max_steps` | `--max-steps` | `None` (full epoch) |
| `mlflow_uri` | `--mlflow-uri` | `"http://localhost:5000"` |
| `run_name` | `--run-name` | `"pa-detector-run"` |

### Logged Metrics (per epoch, `step=epoch`)

| Metric | Description |
|---|---|
| `train_loss` | UncertaintyWeightedLoss total value for the epoch |
| `val_macro_f1` | Macro-averaged F1 across all 5 tone classes (sklearn) |
| `pa_mae` | Mean absolute error for PA regression head |
| `sarcasm_mae` | Mean absolute error for sarcasm regression head |

### Run Tags

| Tag | Source |
|---|---|
| `git_sha` | `git rev-parse --short HEAD` (falls back to `"unknown"`) |
| `dataset_hash` | SHA-256 of first 65536 bytes of `train.parquet`, 12 hex chars |
| `hardware` | `args.device` value — `"cpu"` or `"cuda"` |

### Artifacts

| Artifact | Logging call | Notes |
|---|---|---|
| `model/` (PyTorch MLflow flavour) | `mlflow.pytorch.log_model(model, "model", registered_model_name="pa-detector")` | Registers/updates the `pa-detector` model in the MLflow Model Registry |
| `eval.json` | `mlflow.log_artifact("eval.json")` | Keys: `macro_f1`, `pa_mae`, `sarcasm_mae`, `per_class_f1` |

The model is served from the **Production** stage alias:
`models:/pa-detector/Production`.

---

## 8. Drift Detection Implementation

**Source:** `backend/app/services/drift.py` → `class DriftMonitor`

### Algorithm

1. **Startup** — load `data/reference/feature_stats.json` from the `drift_baseline`
   DVC stage. Extract `length_quantiles` list as the reference distribution.
2. **Per request** — `DriftMonitor.update(text: str)`:
   - Append `float(len(text))` to `_length_window` (a `collections.deque(maxlen=200)`).
   - If `len(window) >= 30` and `len(reference) >= 10`, run:
     `scipy.stats.ks_2samp(list(window), reference)` → `(stat, pvalue)`.
   - Return `{"ks_pvalue": pvalue, "oov_rate": 0.0}`.
   - `ks_pvalue = 1.0` before 30 samples or on any exception (fail-open design).
3. **Metric export** — predict route writes `ks_pvalue` to
   `drift_input_length_ks_pvalue` Gauge and `oov_rate` to `drift_oov_rate` Gauge.
4. **Alert** — Prometheus fires `DriftDetected` when
   `drift_input_length_ks_pvalue < 0.01` for 5 consecutive minutes.

### Reference Baseline Computation

**Source:** `src/data/drift_baseline.py` → `compute_baseline(df)`

| Output key | Computation |
|---|---|
| `length_mean` | `np.mean(text.str.len())` |
| `length_std` | `np.std(text.str.len())` |
| `length_quantiles` | `{str(q): np.quantile(lengths, q)}` for q in `{0.1, 0.25, 0.5, 0.75, 0.9, 0.99}` |
| `vocab` | Top-10,000 tokens by frequency from `text.lower().split()` |

Written to `data/reference/feature_stats.json` by the `drift_baseline` DVC stage.
The quantile values in `length_quantiles` form the reference sample for the KS test.

---

## 9. Key Modules

| File | Class / Function | Responsibility |
|---|---|---|
| `src/data/ingest.py` | `main()` | Iterates `ALL_ADAPTERS`, calls `adapter.load()`, caches to `data/raw/` |
| `src/data/sources.py` | `SourceAdapter` (ABC), `SarcasmHeadlinesAdapter`, `GoEmotionsAdapter`, `ISarcasmAdapter`, `EnronSubsetAdapter` | Per-dataset download and caching; returns DataFrame with `text`, source columns |
| `src/data/clean.py` | `clean_text()`, `clean_dataframe()` | URL/email masking (`<URL>`, `<EMAIL>`), whitespace normalisation, length filter (`min_len=5`) |
| `src/data/label_map.py` | `to_unified()` | Maps source-specific schemas to unified `[text, passive_aggression, sarcasm, tone, source, weak_label]` |
| `src/data/drift_baseline.py` | `compute_baseline()` | Computes reference stats from processed data for KS drift test |
| `src/data/synthesize.py` | `SyntheticGenerator`, `GeneratedSample` | Ollama/Gemma3 few-shot PA email synthesis; validated via Pydantic; not wired to Airflow DAG |
| `src/features/tokenize.py` | `UnifiedDataset`, `get_tokenizer()` | `torch.utils.data.Dataset` wrapping processed parquet; returns `input_ids`, `attention_mask`, and three label tensors |
| `src/models/multitask.py` | `PassiveAggressiveDetector`, `MultiTaskOutput` | DistilBERT + 3 linear heads; single forward pass returning 4-field dataclass |
| `src/models/loss.py` | `UncertaintyWeightedLoss` | Kendall & Gal multi-task uncertainty weighting; 3 learned `log_sigma` parameters |
| `src/train.py` | `main()`, `train_one_epoch()`, `evaluate_epoch()` | MLflow run; trains model; logs all params, metrics, tags, and artefacts |
| `src/evaluate.py` | `compute_metrics()`, `main()` | Reads `models/predictions.parquet`; computes `macro_f1`, `pa_mae`, `sarcasm_mae`, `per_class_f1`; writes `eval.json` |
| `backend/app/main.py` | `app` (FastAPI), `lifespan` | App bootstrap; DB engine, model client, drift monitor init; Prometheus middleware |
| `backend/app/api/predict.py` | `predict_email()` | Calls model client, persists DB row, updates 5 drift/inference metrics, builds highlighted phrases |
| `backend/app/api/feedback.py` | `submit_feedback()` | Updates `user_feedback` column; increments `feedback_votes_total`; returns 204 |
| `backend/app/api/health.py` | `health()`, `ready()`, `set_ready()` | Liveness / readiness probes; `_ready` flag toggled by lifespan |
| `backend/app/api/admin.py` | `receive_alert()` | Logs Alertmanager webhook body via loguru; returns `{"status": "received"}` |
| `backend/app/db/models.py` | `Prediction` (SQLAlchemy ORM) | 11-column predictions table; PK is UUID4 string |
| `backend/app/db/session.py` | `get_engine()`, `get_db()` | Engine factory (Postgres if `POSTGRES_HOST` set, else SQLite); session generator |
| `backend/app/services/drift.py` | `DriftMonitor` | Rolling 200-sample KS test; fail-open (returns 1.0) before 30 samples |
| `backend/app/services/highlighter.py` | `attributions_to_highlighted_phrases()`, `highlight_spans()` | Converts token attribution dicts to merged character-offset spans; threshold=0.5 |
| `backend/app/services/model_client.py` | `MockModelClient`, `HTTPModelClient`, `ModelClient` (Protocol) | Predict interface; mock for dev/test uses PA phrase heuristics; HTTP calls `/invocations` |
| `backend/app/observability/metrics.py` | 8 `prometheus_client` objects | Counter/Histogram/Gauge definitions — all 8 metrics instrumented here |
| `backend/app/config.py` | `Settings` (Pydantic BaseSettings) | All env-var configuration with defaults; loaded once via `get_settings()` |
| `frontend/app.py` | Streamlit app (`st.*` calls) | Single-page UI; `BACKEND_URL` env var; calls `/predict` and `/feedback`; sidebar pipeline links |
| `frontend/helpers.py` | `build_highlight_html()`, `tone_to_color()`, `score_to_hex_color()` | HTML highlight rendering; severity score → `#rrggbb` colour; tone → CSS colour name |
| `contracts/tone_enum.py` | `Tone` (str Enum) | Canonical tone enum shared by backend schemas and training label map |
| `contracts/schemas.json` | — | JSON Schema for API contracts; used by `scripts/export_schemas.py` |
| `airflow/dags/training_pipeline.py` | `dag` (Airflow DAG) | 6-task daily pipeline; `BashOperator` tasks mirror DVC stage commands |
| `monitoring/prometheus/alert_rules.yml` | — | `HighErrorRate` (5xx > 5% / 2 min) and `DriftDetected` (KS < 0.01 / 5 min) |
| `monitoring/grafana/dashboards/api-health.json` | — | API Health dashboard: request rate, p95 latency; 10s refresh |
| `monitoring/grafana/dashboards/ml-health.json` | — | ML Health dashboard: inference p95, drift KS p-value, OOV rate, feedback counts; 10s refresh |
