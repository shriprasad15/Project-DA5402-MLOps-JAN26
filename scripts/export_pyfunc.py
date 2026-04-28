"""Export the best registered pa-detector pytorch version to a CPU pyfunc model.

The pyfunc wrapper handles tokenization internally so the model-server
accepts raw text and returns structured predictions.

Usage:
    python scripts/export_pyfunc.py                          # auto-picks best pytorch version
    python scripts/export_pyfunc.py --version 3              # specific version
    python scripts/export_pyfunc.py --mlflow-uri http://localhost:5000
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import pandas as pd
import torch

TONE_LABELS = ["neutral", "friendly", "assertive", "aggressive", "passive_aggressive"]


class PADetectorPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from transformers import AutoTokenizer
        from src.models.multitask import PassiveAggressiveDetector  # noqa: F401

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = torch.load(
            context.artifacts["model_path"], map_location="cpu", weights_only=False
        )
        self.model.eval()
        self.tone_labels = TONE_LABELS

    def predict(self, context, model_input):
        texts = (
            model_input["text"].tolist()
            if isinstance(model_input, pd.DataFrame)
            else list(model_input)
        )
        results = []
        with torch.no_grad():
            for text in texts:
                enc = self.tokenizer(
                    str(text),
                    max_length=128,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                out = self.model(enc["input_ids"], enc["attention_mask"])
                pa = float(torch.sigmoid(out.pa_logits).squeeze())
                sarc = float(torch.sigmoid(out.sarcasm_logits).squeeze())
                tp = torch.softmax(out.tone_logits, dim=-1).squeeze()
                ti = int(tp.argmax())
                results.append(
                    {
                        "pa_score": round(pa, 4),
                        "sarcasm_score": round(sarc, 4),
                        "tone": self.tone_labels[ti],
                        "tone_confidence": round(float(tp[ti]), 4),
                        "hidden": out.hidden.squeeze().tolist(),
                    }
                )
        return pd.DataFrame(results)


def _is_pytorch_version(client: mlflow.tracking.MlflowClient, v) -> bool:
    """Return True if this version is a raw pytorch model (not a pyfunc CPU export)."""
    try:
        run = client.get_run(v.run_id)
        # pyfunc export runs are named "pyfunc-cpu-*"
        return not run.info.run_name.startswith("pyfunc-cpu")
    except Exception:
        return False


def _best_pytorch_version(client: mlflow.tracking.MlflowClient) -> str:
    """Return the version number of the pytorch model with the highest best_val_macro_f1."""
    versions = client.search_model_versions("name='pa-detector'")
    pytorch_versions = [v for v in versions if _is_pytorch_version(client, v)]

    if not pytorch_versions:
        raise RuntimeError("No pytorch versions found in model registry. Run training first.")

    def f1_for(v):
        try:
            return client.get_run(v.run_id).data.metrics.get("best_val_macro_f1", 0.0)
        except Exception:
            return 0.0

    best = max(pytorch_versions, key=f1_for)
    f1 = f1_for(best)
    print(f"Best pytorch version: v{best.version} (best_val_macro_f1={f1:.4f})")
    return best.version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", type=str, default=None,
        help="Specific pytorch model registry version to export. Default: best by val_macro_f1.",
    )
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("pa-detector")

    client = mlflow.tracking.MlflowClient()

    version_str = args.version if args.version else _best_pytorch_version(client)

    v = client.get_model_version("pa-detector", version_str)
    print(f"Exporting pa-detector v{version_str} (run {v.run_id[:8]}...)")

    tmp = tempfile.mkdtemp()
    print("Downloading model weights → CPU...")
    local_path = mlflow.artifacts.download_artifacts(
        f"runs:/{v.run_id}/model/data/model.pth", dst_path=tmp
    )
    model_cpu = mlflow.pytorch.load_model(f"runs:/{v.run_id}/model", map_location="cpu")
    model_cpu.eval()
    print(f"Loaded {type(model_cpu).__name__} to CPU")

    # Carry over the val metrics from the source run so the pyfunc version is
    # also queryable by f1 score in the registry.
    source_metrics = client.get_run(v.run_id).data.metrics

    repo_root = str(Path(__file__).resolve().parents[1])
    with mlflow.start_run(run_name=f"pyfunc-cpu-v{version_str}"):
        # Log same metrics so this run is comparable in the UI
        for k, val in source_metrics.items():
            mlflow.log_metric(k, val)
        mlflow.log_param("source_version", version_str)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=PADetectorPyfunc(),
            artifacts={"model_path": local_path},
            registered_model_name="pa-detector",
            code_paths=[repo_root],
            pip_requirements=["torch==2.4.1", "transformers==4.44.2", "pandas==2.2.3"],
        )

    all_versions = client.search_model_versions("name='pa-detector'")
    new_version = sorted(all_versions, key=lambda x: int(x.version))[-1].version
    print(
        f"\nDone — pa-detector v{new_version} registered (CPU pyfunc, ready for model-server)"
    )

    # Tag as champion alias — docker-compose always uses models:/pa-detector@champion
    client.set_registered_model_alias("pa-detector", "champion", new_version)
    print(f"Alias 'champion' → pa-detector v{new_version}")


if __name__ == "__main__":
    main()
