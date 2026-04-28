# Test Plan — Passive-Aggressive Email Detector (DA5402 Wave 3)

## 1. Acceptance Criteria

All of the following criteria must be met before a release is considered complete:

| ID | Criterion | Verification method |
|---|---|---|
| AC-1 | All unit tests pass | `pytest -m "unit"` exits 0 |
| AC-2 | Backend API responds correctly to all 6 endpoints | `backend/tests/test_api.py` with FastAPI `TestClient` |
| AC-3 | Model forward pass produces correct output tensor shapes | `tests/models/test_multitask.py::test_forward_shapes` |
| AC-4 | Feedback loop records votes in the database | `test_feedback_valid` — POST `/predict` then POST `/feedback` → 204 |
| AC-5 | Drift monitor returns a valid dict with expected keys | `tests/test_drift.py::test_returns_dict_keys` |
| AC-6 | Frontend helpers produce correct HTML output | `frontend/tests/test_format.py` |
| AC-7 | Schema contracts are upheld | `tests/test_schemas.py` — Tone enum, length constraints, vote values |
| AC-8 | Evaluate module returns all expected metric keys | `tests/test_evaluate.py::test_expected_keys` |

---

## 2. Test Categories

| Category | Marker | Description | Requires |
|---|---|---|---|
| Unit | `@pytest.mark.unit` | Fast, fully isolated, no network or Docker | conda env `mlops` only |
| Slow | `@pytest.mark.slow` | > 5 s; mocked DistilBERT training smoke test | conda env `mlops` only |
| Integration | `@pytest.mark.integration` | Requires live Docker Compose stack | `docker compose up -d --wait` |
| E2E | `@pytest.mark.e2e` | Full browser flow via Playwright | Not yet implemented (see `docs/future-work.md`) |

```bash
# CI default — unit tests only
pytest -m "not slow and not integration" -q

# Slow tests
pytest -m slow -q

# Integration tests (requires running stack)
pytest -m integration -q

# All non-integration tests
pytest -m "not integration" -q
```

---

## 3. Test Tools

| Tool | Version | Purpose |
|---|---|---|
| pytest | 8.3.3 | Test runner, marker system, fixtures |
| pytest-cov | latest | Coverage reporting (`--cov=src --cov=backend/app`) |
| pytest-mock | latest | `mocker` fixture for patching |
| `unittest.mock` | stdlib | `patch`, `MagicMock` — used for DistilBERT and Ollama mocking |
| `fastapi.testclient.TestClient` | (from FastAPI) | In-process ASGI test client; no server process needed |
| hypothesis | latest | Property-based testing; available in dev requirements |

Test configuration is in `pyproject.toml` `[tool.pytest.ini_options]`:

- `testpaths = ["tests", "backend/tests", "src", "frontend/tests"]`
- `--strict-markers` — unregistered markers raise an error
- `addopts = "-ra --strict-markers --strict-config"`

---

## 4. Test Cases

### 4.1 Data Pipeline Tests

| Test ID | File | Test Name | Type | Expected Result |
|---|---|---|---|---|
| T01a | `tests/data/test_clean.py` | `test_clean_text_strips_urls_and_emails` | unit | `"hello http://x.com and me@me.com"` → `"hello <URL> and <EMAIL>"` |
| T01b | `tests/data/test_clean.py` | `test_clean_text_collapses_whitespace` | unit | `"a   b\n\n\tc"` → `"a b c"` |
| T01c | `tests/data/test_clean.py` | `test_clean_dataframe_drops_rows_below_min_length` | unit | Rows with `len(text) < min_len` are filtered; result has 1 row |
| T02a | `tests/data/test_label_map.py` | `test_sarcasm_headlines_maps_to_unified_schema` | unit | Output columns equal `UNIFIED_COLS` exactly; `sarcasm=1.0` preserved; `weak_label=True` |
| T02b | `tests/data/test_label_map.py` | `test_goemotions_anger_maps_to_aggressive_tone` | unit | GoEmotions label index 2 (anger) → `Tone.AGGRESSIVE.value` |
| T03 | `tests/data/test_drift_baseline.py` | `test_baseline_has_expected_fields` | unit | `compute_baseline()` dict has `length_mean`, `length_std`, `length_quantiles`, `vocab`; `"0.5"` key present in quantiles; `length_mean > 0` |

### 4.2 Model Tests

| Test ID | File | Test Name | Type | Expected Result |
|---|---|---|---|---|
| T04 | `tests/models/test_multitask.py` | `test_forward_shapes` | unit | `pa_logits` → `(2,1)`, `sarcasm_logits` → `(2,1)`, `tone_logits` → `(2,5)`, `hidden` → `(2,768)` |
| T05 | `tests/models/test_multitask.py` | `test_quantized_default_false` | unit | `model.quantized is False` |
| T06 | `tests/models/test_loss.py` | `test_loss_positive` | unit | `UncertaintyWeightedLoss([0.5, 0.3, 0.7]).item() > 0` |
| T07 | `tests/models/test_loss.py` | `test_no_nans` | unit | `torch.isnan(loss)` is `False` |

Both model tests mock `DistilBertModel.from_pretrained` to avoid downloading weights
during unit test runs. The mock returns a `BaseModelOutput` with `last_hidden_state`
of zeros shape `[2, 16, 768]`.

### 4.3 Backend API Tests

| Test ID | File | Test Name | Type | Expected Result |
|---|---|---|---|---|
| T08 | `backend/tests/test_api.py` | `test_health` | unit | `GET /health` → 200, `{"status": "ok"}` |
| T09 | `backend/tests/test_api.py` | `test_ready_before_startup` | unit | `GET /ready` → 200 or 503 (both valid depending on lifespan context) |
| T10 | `backend/tests/test_api.py` | `test_predict_happy_path` | unit | `POST /predict {"text": "as per my last email..."}` → 200; `prediction_id`, `scores`, `tone`, `highlighted_phrases` all present |
| T11 | `backend/tests/test_api.py` | `test_predict_empty_text` | unit | `POST /predict {"text": ""}` → 422 |
| T12 | `backend/tests/test_api.py` | `test_predict_whitespace_only` | unit | `POST /predict {"text": "   "}` → 422 |
| T13 | `backend/tests/test_api.py` | `test_feedback_valid` | unit | POST `/predict` → extract `prediction_id` → POST `/feedback {"vote": "up"}` → 204 |
| T14 | `backend/tests/test_api.py` | `test_feedback_invalid_vote` | unit | `POST /feedback {"vote": "meh"}` → 422 |
| T15 | `backend/tests/integration/test_http_client.py` | `test_http_client_*` | integration | Live model server calls via `HTTPModelClient` (requires `docker compose up`) |

All unit backend tests use `fastapi.testclient.TestClient(app)` as a context manager,
which triggers the lifespan startup hook (initialising SQLite engine, `MockModelClient`,
and `DriftMonitor`) before each test.

### 4.4 Drift Monitor Tests

| Test ID | File | Test Name | Type | Expected Result |
|---|---|---|---|---|
| T16 | `tests/test_drift.py` | `test_returns_dict_keys` | unit | `DriftMonitor(None).update("hello world")` returns dict with keys `ks_pvalue` and `oov_rate` |
| T17 | `tests/test_drift.py` | `test_in_distribution` | unit | 100 updates with identical text and no reference data → `ks_pvalue == 1.0` (fail-open) |

### 4.5 Schema Tests

| Test ID | File | Test Name | Type | Expected Result |
|---|---|---|---|---|
| T18 | `tests/test_schemas.py` | `test_tone_enum_has_expected_values` | unit | `{t.value for t in Tone}` equals `{"neutral", "friendly", "assertive", "aggressive", "passive_aggressive"}` exactly |
| T19 | `tests/test_schemas.py` | `test_predict_request_rejects_empty_text` | unit | `PredictRequest(text="")` raises `ValueError` |
| T20 | `tests/test_schemas.py` | `test_predict_request_rejects_text_over_5000_chars` | unit | `PredictRequest(text="x"*5001)` raises `ValueError` |
| T21 | `tests/test_schemas.py` | `test_predict_response_scores_clamped_0_1` | unit | Valid `PredictResponse` constructed without error; score value preserved |
| T22 | `tests/test_schemas.py` | `test_feedback_request_accepts_only_up_or_down` | unit | `FeedbackRequest(prediction_id="abc", vote="meh")` raises `ValueError` |

### 4.6 Evaluate Tests

| Test ID | File | Test Name | Type | Expected Result |
|---|---|---|---|---|
| T23 | `tests/test_evaluate.py` | `test_expected_keys` | unit | `compute_metrics(df)` returns dict with keys `macro_f1`, `pa_mae`, `sarcasm_mae`, `per_class_f1` |

### 4.7 Frontend Tests

| Test ID | File | Test Name | Type | Expected Result |
|---|---|---|---|---|
| T24 | `frontend/tests/test_format.py` | `test_tone_to_color_passive_aggressive` | unit | `tone_to_color("passive_aggressive")` → `"red"` |
| T25 | `frontend/tests/test_format.py` | `test_tone_to_color_unknown` | unit | `tone_to_color("xyz")` → `"grey"` |
| T26 | `frontend/tests/test_format.py` | `test_score_to_hex_zero` | unit | `score_to_hex_color(0.0)` → `"#ffc8c8"` |
| T27 | `frontend/tests/test_format.py` | `test_score_to_hex_one` | unit | `score_to_hex_color(1.0)` → `"#ff0000"` |
| T28 | `frontend/tests/test_format.py` | `test_build_highlight_html_single` | unit | Single highlight `[{start:6, end:11, severity:1.0}]` over `"hello world"` → HTML with `<mark>` tag wrapping `"world"` |
| T29 | `frontend/tests/test_format.py` | `test_build_highlight_html_empty` | unit | `build_highlight_html("hello", [])` → `"hello"` (original text unchanged) |

### 4.8 Smoke and Infrastructure Tests

| Test ID | File | Test Name | Type | Expected Result |
|---|---|---|---|---|
| T30 | `tests/test_smoke.py` | `test_repo_has_readme` | unit | `README.md` exists at repo root |
| T31 | `tests/test_smoke.py` | `test_repo_has_openapi` | unit | `openapi.yaml` exists at repo root |
| T32 | `tests/test_smoke.py` | `test_repo_has_pyproject` | unit | `pyproject.toml` exists at repo root |
| T33 | `tests/airflow/test_dag.py` | `test_dag_importable` | unit | `from airflow.dags.training_pipeline import dag` succeeds (skipped if airflow not installed) |
| T34 | `tests/airflow/test_dag.py` | `test_dag_id` | unit | `dag.dag_id == "pa_training_pipeline"` |
| T35 | `tests/airflow/test_dag.py` | `test_dag_task_count` | unit | `len(dag.tasks) == 6` |

### 4.9 Slow Tests

| Test ID | File | Test Name | Type | Expected Result |
|---|---|---|---|---|
| T36 | `tests/test_train_smoke.py` | `test_train_one_epoch_smoke` | slow | `train_one_epoch()` completes with mocked DistilBERT and mocked tokenizer; returned loss > 0 |

---

## 5. Coverage Targets

| Module | Target line coverage |
|---|---|
| `src/` | ≥ 70% |
| `backend/app/` | ≥ 80% |
| `frontend/` | ≥ 60% |

Generate coverage report:

```bash
pytest --cov=src --cov=backend/app --cov=frontend \
       -q -m "not slow and not integration" \
       --cov-report=term-missing
```

---

## 6. CI Configuration

CI is defined in `.github/workflows/ci.yml`. On every push and pull request:

1. Checkout repository
2. Set up Python 3.11 (explicit version match for conda environment)
3. Install dependencies: `pip install -r requirements-dev.txt`
4. Run linting: `ruff check .`
5. Run type checking: `mypy src/ backend/app/`
6. Run unit tests: `pytest -m "not slow and not integration" -q`

Slow and integration tests are run manually or on scheduled CI runs. Integration
tests require `docker compose up -d --wait --wait-timeout 180` before the test
command.
