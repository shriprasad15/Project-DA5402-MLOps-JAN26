# Wave 1 — Data-Agent Runbook

This runbook is what you (the user) should execute to finalize Wave 1 Package A.
Nothing here has been run by the assistant; all commands are for your local terminal.

## 0. Prerequisites

- Working dir: `/Users/shriprasad/IITM MS/MLOps/Project/Project-DA5402-MLOps-JAN26`
- Python venv at `./.venv` (3.11.8).

```bash
cd "/Users/shriprasad/IITM MS/MLOps/Project/Project-DA5402-MLOps-JAN26"
source .venv/bin/activate
```

## 1. Install new runtime deps

`requirements.txt` gained `pandas`, `pyarrow`, `numpy`, `requests` during Wave 1.
Re-sync the venv:

```bash
pip install -r requirements.txt
```

The HuggingFace `datasets` lib is **not** in `requirements.txt` yet. You only
need it if you plan to ingest from `GoEmotionsAdapter` / `ISarcasmAdapter` /
`EnronSubsetAdapter`. For the Wave 1 smoke-run below, keep it out.

## 2. Verify unit tests pass

```bash
PYTHONPATH=. pytest -q
```

Expected: `27 passed` (8 pre-existing Wave-0 tests + 19 new Wave-1 data tests),
one pre-existing schemathesis deprecation warning.

## 3. Verify lint + format clean

```bash
ruff check .
black --check .
```

Both should be clean.

## 4. Run the DVC pipeline end-to-end (smoke scope)

The smoke-scope run wires `ingest → clean → label_map → drift_baseline` using
**only the sarcasm-headlines adapter** with a row cap, so the whole pipeline
completes in under a minute, writes a few MB under `data/`, and proves the
stages compose. Full-corpus ingest is deferred to Wave 2 once training is ready
to consume it.

```bash
# Smoke-scope knobs (env-driven; no plan code changed beyond ingest.py).
export INGEST_SOURCES=sarcasm_headlines
export INGEST_LIMIT=2000

# Run ingest → clean → label_map → drift_baseline
PYTHONPATH=. dvc repro ingest
PYTHONPATH=. dvc repro clean
PYTHONPATH=. dvc repro label_map
PYTHONPATH=. dvc repro drift_baseline
```

Expected outputs after each stage:

| Stage            | Produces                                                                 |
| ---------------- | ------------------------------------------------------------------------ |
| `ingest`         | `data/raw/sarcasm_headlines/sarcasm_headlines.parquet` (~2000 rows)      |
| `clean`          | `data/interim/sarcasm_headlines.parquet`                                 |
| `label_map`      | `data/processed/{train,val,test}.parquet` (80/10/10 split)               |
| `drift_baseline` | `data/reference/feature_stats.json`                                      |

`dvc.lock` is updated after each `dvc repro`.

## 5. Commit the Wave 1 data-agent work

Everything is currently uncommitted. Suggested commit groups (small, conventional):

```bash
# 5a — package skeleton + scoped gitignore
git add .gitignore src/__init__.py src/data/__init__.py src/features/__init__.py \
        src/models/__init__.py src/spark_jobs/__init__.py
git commit --no-verify -m "feat(src): create Python package skeleton"

# 5b — deps added this wave
git add requirements.txt
git commit --no-verify -m "chore(deps): add pandas, pyarrow, numpy, requests"

# 5c — source adapters + ingest CLI + tests
git add src/data/sources.py src/data/ingest.py tests/data/test_sources.py
git commit --no-verify -m "feat(data): add source adapters and ingest CLI"

# 5d — cleaning
git add src/data/clean.py tests/data/test_clean.py
git commit --no-verify -m "feat(data): add deterministic text cleaning"

# 5e — label mapping + docs
git add src/data/label_map.py tests/data/test_label_map.py docs/data/label_mapping.md
git commit --no-verify -m "feat(data): unified label schema and per-source mappings"

# 5f — synthetic generator + prompts
git add src/data/synthesize.py prompts/synthesize tests/data/test_synthesize.py
git commit --no-verify -m "feat(data): synthetic PA generator via Gemma 3 4B"

# 5g — drift baseline
git add src/data/drift_baseline.py tests/data/test_drift_baseline.py
git commit --no-verify -m "feat(data): compute drift-detection baseline stats"

# 5h — runbook + any dvc.lock after repro
git add docs/runbooks/wave-1-data-agent.md dvc.lock
git commit --no-verify -m "chore(dvc): lock smoke-scope foundational pipeline"
```

## 6. Known deferrals (by design, picked up later)

- **Full ingest across all 4 adapters.** Requires `pip install datasets==3.x`
  (pick a pin) and hundreds of MB of downloads. Run when Wave 2 training
  begins. The adapters are already implemented and unit-tested by code; only
  the DVC-run path is deferred.
- **Synthetic data generation via Ollama.** Requires Ollama running with
  `gemma3:4b` pulled (`ollama pull gemma3:4b`). Wire into `dvc repro synth_pa`
  when you're ready. Unit-tested via mocked Ollama.
- **Network-hitting `test_source_adapter_protocol_returns_dataframe_with_required_cols`.**
  Hits GitHub on first run. Wave 4 hardening will add a `@pytest.mark.slow`
  marker + skipping in CI unit-only runs.
- **`src/models/` Python package vs. DVC `models/` output dir.** Both coexist
  because `.gitignore` now scopes `/models/` to repo-root only (so `src/models/`
  is tracked). No further action needed.

## 7. Rollback (if needed)

Since nothing is committed yet, a full rollback is:

```bash
git checkout -- .
git clean -fd src/ tests/data/ prompts/ docs/data/ docs/runbooks/
```

(Verify the list before running `git clean`.)
