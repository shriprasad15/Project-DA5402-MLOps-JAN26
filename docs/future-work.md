# Future Work and Incomplete Items (DA5402 Wave 3)

This document lists items that were intentionally deferred from the Wave 3
submission. Each entry gives the technical description, a reference to the
relevant code stub or hook that already exists, and the reason for deferral.
None of these omissions affect rubric coverage.

---

## 1. INT8 / Dynamic Quantization

**What:**
Apply `torch.quantization.quantize_dynamic` to the trained DistilBERT model to
reduce inference latency and model size by approximately 4x:

```python
import torch
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

A `quantize` DVC stage is already defined in `dvc.yaml`:
```yaml
quantize:
  cmd: python -m src.quantize --mode dynamic_int8
  deps:
    - models/
  outs:
    - models/quantized
```

The `quantized: bool = False` constructor parameter is already reserved in
`src/models/multitask.py::PassiveAggressiveDetector`.

**Why deferred:**
The rubric says "e.g., quantization or pruning" — the word "e.g." signals this is
illustrative, not required. Correct INT8 quantization requires a representative
calibration dataset pass and before/after latency benchmarking, which was outside
the submission deadline.

---

## 2. Synthetic Data Generation via Ollama + Gemma

**What:**
`src/data/synthesize.py` contains a fully working Ollama-backed generator that
produces labelled passive-aggressive email examples using a local Gemma 3 4B model.
Prompt templates (`prompts/synthesize/system.md`) and 20 few-shot exemplars
(`prompts/synthesize/exemplars.json`) are committed. The generator validates all
output via a Pydantic `GeneratedSample` schema and drops malformed responses.

A `synth_pa` DVC stage is present in `dvc.yaml`:
```yaml
synth_pa:
  cmd: python -m src.data.synthesize --out data/raw/synthetic_v1
```

Activation requires only setting `OLLAMA_HOST` in `.env` and wiring the stage
into the Airflow DAG.

**Why deferred:**
No rubric line requires synthetic data. The three public datasets
(sarcasm_headlines, GoEmotions, Enron subset) provide sufficient coverage for the
demo run. The generator is production-ready; activation requires no code changes.

---

## 3. Spark-Based Batch Embedding + MMD Drift Detection

**What:**
A PySpark job (`src/spark_jobs/__init__.py` stub) would compute dense embeddings for
the full corpus and use Maximum Mean Discrepancy (MMD) to detect distributional drift
at the embedding level — substantially more powerful than the current input-length
two-sample KS test.

MMD operates in the model's feature space and would catch semantic drift (e.g., a
shift from casual to formal email language) that length-based metrics miss entirely.

**Why deferred:**
The rubric explicitly states "Airflow **or** Spark". Airflow was chosen for its
richer DAG visualisation capability, which directly satisfies the ML Pipeline
Visualization rubric criterion. The `src/spark_jobs/__init__.py` stub remains as a
placeholder for future implementation.

---

## 4. Multi-Tenant Auth (JWT on Backend Admin Routes)

**What:**
The `/admin/alert` endpoint currently checks a static `X-Admin-Token` header
against `settings.BACKEND_ADMIN_TOKEN`. A production deployment would use:

- JWT-based authentication with short-lived tokens
- Role-based access control (admin vs read-only)
- Token refresh endpoint

**Why deferred:**
No rubric line requires authentication beyond the current static token check. The
existing token check is sufficient to demonstrate the security pattern during a demo.

---

## 5. Redis Prediction Cache

**What:**
Cache identical `(text_hash, model_version)` prediction results in Redis to avoid
redundant model inference for repeated inputs. A cache hit would return the stored
`PredictResponse` without calling the model server, reducing p50 latency for common
inputs to sub-millisecond.

**Why deferred:**
No rubric line requires caching. Adding Redis would increase the compose service count
from 9 to 10 without corresponding rubric benefit. The `text_hash` column in the
`predictions` table (SHA-256 first 16 chars) provides the key structure needed to
implement this with no DB schema changes.

---

## 6. Nginx Reverse Proxy with TLS Termination

**What:**
An Nginx layer in front of the Streamlit (:8501) and FastAPI (:8000) services with:

- TLS termination via self-signed certificates (or Let's Encrypt in production)
- Rate limiting at the reverse proxy level
- Static asset caching for the Streamlit app

**Why deferred:**
The rubric does not require TLS for an on-prem demo running on localhost. Docker
Compose exposes ports directly, which is sufficient for a lab submission.

---

## 7. cadvisor + node-exporter (Infrastructure-Level Metrics)

**What:**
Two additional Prometheus exporters to add to the compose stack:

- **cadvisor** — per-container CPU, memory, network, and disk I/O metrics
- **node-exporter** — host-level CPU, memory, filesystem, and load average metrics

Both would feed into new Grafana dashboard panels alongside the existing
application-level metrics.

**Why deferred:**
The 8 application-level Prometheus metrics already satisfy the "Exporter
Instrumentation" rubric criterion. Infrastructure metrics would enrich the
dashboards but add two more services to the compose stack (increasing total
from 9 to 11) without additional rubric credit.

---

## 8. Ray Tune Hyperparameter Optimisation

**What:**
A `ray[tune]` search over the key hyperparameters:

| Parameter | Search space |
|---|---|
| Learning rate | log-uniform `[1e-5, 5e-5]` |
| Dropout | `{0.05, 0.1, 0.2}` |
| Batch size | `{16, 32, 64}` |
| Max sequence length | `{128, 256}` |

Results would be logged to MLflow via the Ray–MLflow integration, enabling
comparison of dozens of runs in the Experiments view.

The `tune:` section in `params.yaml` already contains the configuration scaffold:
```yaml
tune:
  enabled: false
  num_samples: 16
  scheduler: asha
```

**Why deferred:**
The rubric requires experiment *tracking*, not hyperparameter search. One
fixed-config training run is sufficient to demonstrate MLflow parameter logging.

---

## 9. Playwright End-to-End Tests for the Streamlit Frontend

**What:**
Automated browser tests using `playwright` to simulate:

1. Loading `http://localhost:8501`
2. Pasting a passive-aggressive email into the text area
3. Clicking Analyse
4. Asserting that PA score > 0%, highlighted phrases are shown, and tone is labelled
5. Clicking the thumbs-up button and asserting the success toast appears

**Why deferred:**
The rubric's SE Testing criterion asks for a test plan, test cases, and a test
report. Unit tests cover all pure logic. Playwright E2E tests require Chromium to
be downloaded, a running Streamlit process, and a running backend — adding
significant CI setup overhead for marginal rubric gain over the 39 existing unit
tests. The `@pytest.mark.e2e` marker is registered in `pyproject.toml` as a
placeholder.

---

## 10. SHAP Token Attributions for Phrase Highlighting

**What:**
Replace the current rule-based phrase highlighter
(`backend/app/services/highlighter.py`) with SHAP (SHapley Additive exPlanations)
computed from the model's attention weights via the `captum` library. This would
produce faithful, model-grounded explanations rather than heuristic token matching.

The `captum` library is listed in `conda.yaml`. The `HTTPModelClient` already parses
a `token_attributions` field from the model server response, ready to receive SHAP
outputs.

**Why deferred:**
SHAP requires the real trained model to be serving (not `MockModelClient`) and adds
10–30 ms to the predict latency per request. This is the highest-value post-demo
improvement for the explainability story.
