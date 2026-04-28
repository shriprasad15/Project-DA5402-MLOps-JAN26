from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import pandas as pd
import torch
from loguru import logger
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.features.tokenize import UnifiedDataset, get_tokenizer
from src.models.loss import UncertaintyWeightedLoss
from src.models.multitask import PassiveAggressiveDetector

TONE_LABELS_SERVING = ["neutral", "friendly", "assertive", "aggressive", "passive_aggressive"]


class PADetectorPyfunc(mlflow.pyfunc.PythonModel):
    """Pyfunc wrapper: accepts raw text, handles tokenisation internally."""

    def load_context(self, context):
        import torch as _torch
        from transformers import AutoTokenizer
        from src.models.multitask import PassiveAggressiveDetector as _Model  # noqa: F401

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = _torch.load(
            context.artifacts["model_path"], map_location="cpu", weights_only=False
        )
        self.model.eval()

    def predict(self, context, model_input):
        import torch as _torch

        texts = (
            model_input["text"].tolist()
            if isinstance(model_input, pd.DataFrame)
            else list(model_input)
        )
        results = []
        with _torch.no_grad():
            for text in texts:
                enc = self.tokenizer(
                    str(text), max_length=128, padding="max_length",
                    truncation=True, return_tensors="pt",
                )
                out = self.model(enc["input_ids"], enc["attention_mask"])
                pa = float(_torch.sigmoid(out.pa_logits).squeeze())
                sarc = float(_torch.sigmoid(out.sarcasm_logits).squeeze())
                tp = _torch.softmax(out.tone_logits, dim=-1).squeeze()
                ti = int(tp.argmax())
                results.append({
                    "pa_score": round(pa, 4),
                    "sarcasm_score": round(sarc, 4),
                    "tone": TONE_LABELS_SERVING[ti],
                    "tone_confidence": round(float(tp[ti]), 4),
                    "hidden": out.hidden.squeeze().tolist(),
                })
        return pd.DataFrame(results)


def get_scheduler(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def get_git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def get_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()[:12]


def build_dataset(parquet_path: Path, tokenizer, max_length: int = 128) -> UnifiedDataset:
    df = pd.read_parquet(parquet_path)
    return UnifiedDataset(df, tokenizer, max_length=max_length)


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    device: str,
    *,
    scaler=None,
    max_steps: int | None = None,
    scheduler=None,
    global_step: int = 0,
) -> tuple[float, int]:
    model.train()
    criterion = UncertaintyWeightedLoss(n_tasks=3).to(device)
    total_loss = 0.0
    steps = 0
    for batch in loader:
        if max_steps is not None and steps >= max_steps:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pa_label = batch["pa_label"].to(device)
        sarcasm_label = batch["sarcasm_label"].to(device)
        tone_label = batch["tone_label"].to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                out = model(input_ids, attention_mask)
                pa_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    out.pa_logits.squeeze(-1), pa_label
                )
                sarcasm_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    out.sarcasm_logits.squeeze(-1), sarcasm_label
                )
                tone_loss = torch.nn.functional.cross_entropy(out.tone_logits, tone_label)
                loss = criterion([pa_loss, sarcasm_loss, tone_loss])
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(input_ids, attention_mask)
            pa_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                out.pa_logits.squeeze(-1), pa_label
            )
            sarcasm_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                out.sarcasm_logits.squeeze(-1), sarcasm_label
            )
            tone_loss = torch.nn.functional.cross_entropy(out.tone_logits, tone_label)
            loss = criterion([pa_loss, sarcasm_loss, tone_loss])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        steps += 1
        global_step += 1

    return total_loss / max(steps, 1), global_step


TONE_LABELS = TONE_LABELS_SERVING


def evaluate_epoch(model, loader: DataLoader, device: str) -> dict:
    from sklearn.metrics import f1_score, accuracy_score

    model.eval()
    criterion = UncertaintyWeightedLoss(n_tasks=3).to(device)
    all_tone_preds: list[int] = []
    all_tone_labels: list[int] = []
    pa_abs_errors: list[float] = []
    sarcasm_abs_errors: list[float] = []
    pa_correct: list[int] = []
    sarcasm_correct: list[int] = []
    total_val_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pa_label = batch["pa_label"].to(device)
            sarcasm_label = batch["sarcasm_label"].to(device)
            tone_label = batch["tone_label"].to(device)

            out = model(input_ids, attention_mask)

            pa_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                out.pa_logits.squeeze(-1), pa_label
            )
            sarcasm_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                out.sarcasm_logits.squeeze(-1), sarcasm_label
            )
            tone_loss = torch.nn.functional.cross_entropy(out.tone_logits, tone_label)
            total_val_loss += criterion([pa_loss, sarcasm_loss, tone_loss]).item()
            steps += 1

            tone_preds = out.tone_logits.argmax(dim=-1).cpu().tolist()
            all_tone_preds.extend(tone_preds)
            all_tone_labels.extend(tone_label.cpu().tolist())

            pa_preds = torch.sigmoid(out.pa_logits).squeeze(-1).cpu()
            sarcasm_preds = torch.sigmoid(out.sarcasm_logits).squeeze(-1).cpu()
            pa_abs_errors.extend((pa_preds - pa_label.cpu()).abs().tolist())
            sarcasm_abs_errors.extend((sarcasm_preds - sarcasm_label.cpu()).abs().tolist())
            pa_correct.extend(((pa_preds > 0.5) == (pa_label.cpu() > 0.5)).int().tolist())
            sarcasm_correct.extend(((sarcasm_preds > 0.5) == (sarcasm_label.cpu() > 0.5)).int().tolist())

    macro_f1 = float(f1_score(all_tone_labels, all_tone_preds, average="macro", zero_division=0))
    per_class_f1 = f1_score(all_tone_labels, all_tone_preds, average=None, zero_division=0)
    accuracy = float(accuracy_score(all_tone_labels, all_tone_preds))
    pa_mae = float(sum(pa_abs_errors) / max(len(pa_abs_errors), 1))
    sarcasm_mae = float(sum(sarcasm_abs_errors) / max(len(sarcasm_abs_errors), 1))
    pa_accuracy = float(sum(pa_correct) / max(len(pa_correct), 1))
    sarcasm_accuracy = float(sum(sarcasm_correct) / max(len(sarcasm_correct), 1))
    val_loss = total_val_loss / max(steps, 1)

    metrics = {
        "val_loss": val_loss,
        "val_macro_f1": macro_f1,
        "val_accuracy": accuracy,
        "pa_mae": pa_mae,
        "pa_accuracy": pa_accuracy,
        "sarcasm_mae": sarcasm_mae,
        "sarcasm_accuracy": sarcasm_accuracy,
    }
    for i, label in enumerate(TONE_LABELS):
        metrics[f"f1_{label}"] = float(per_class_f1[i]) if i < len(per_class_f1) else 0.0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/processed/train.parquet")
    parser.add_argument("--val-path", default="data/processed/val.parquet")
    parser.add_argument("--test-path", default="data/processed/test.parquet")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    parser.add_argument("--run-name", default="pa-detector-run")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("pa-detector")

    tokenizer = get_tokenizer()
    model = PassiveAggressiveDetector().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if args.bf16 and args.device != "cpu" else None

    # Estimate total steps for scheduler (use a placeholder if data not available yet)
    if Path(args.data_path).exists():
        _ds_len = len(pd.read_parquet(args.data_path))
        _steps_per_epoch = max(1, _ds_len // args.batch_size)
    else:
        _steps_per_epoch = 100
    total_steps = args.epochs * _steps_per_epoch
    scheduler = get_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(vars(args))
        mlflow.set_tag("git_sha", get_git_sha())
        mlflow.set_tag("hardware", args.device)
        if Path(args.data_path).exists():
            mlflow.set_tag("dataset_hash", get_file_hash(Path(args.data_path)))

        global_step = 0
        best_f1 = 0.0
        best_state: dict | None = None

        for epoch in range(args.epochs):
            if Path(args.data_path).exists():
                train_ds = build_dataset(Path(args.data_path), tokenizer)
                train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
                loss, global_step = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    args.device,
                    scaler=scaler,
                    max_steps=args.max_steps,
                    scheduler=scheduler,
                    global_step=global_step,
                )
                current_lr = scheduler.get_last_lr()[0] if scheduler else args.lr
                logger.info(f"epoch={epoch} train_loss={loss:.4f} lr={current_lr:.2e}")
                mlflow.log_metric("train_loss", loss, step=epoch)
                mlflow.log_metric("learning_rate", current_lr, step=epoch)

            if Path(args.val_path).exists():
                val_ds = build_dataset(Path(args.val_path), tokenizer)
                val_loader = DataLoader(val_ds, batch_size=args.batch_size)
                val_metrics = evaluate_epoch(model, val_loader, args.device)
                logger.info(f"epoch={epoch} val_metrics={val_metrics}")
                for k, v in val_metrics.items():
                    mlflow.log_metric(k, v, step=epoch)

                if val_metrics["val_macro_f1"] > best_f1:
                    best_f1 = val_metrics["val_macro_f1"]
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    logger.info(f"epoch={epoch} new best f1={best_f1:.4f} — saving checkpoint")

        # Restore best weights before test evaluation and registration
        if best_state is not None:
            model.load_state_dict({k: v.to(args.device) for k, v in best_state.items()})
            logger.info(f"Restored best model (val_macro_f1={best_f1:.4f})")
        mlflow.log_metric("best_val_macro_f1", best_f1)

        # Test set evaluation (uses best-epoch weights)
        if Path(args.test_path).exists():
            test_ds = build_dataset(Path(args.test_path), tokenizer)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size)
            test_metrics = evaluate_epoch(model, test_loader, args.device)
            logger.info(f"test_metrics={test_metrics}")
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)

        # Only register if this run beats every previously registered version
        client = mlflow.tracking.MlflowClient()
        all_versions = client.search_model_versions("name='pa-detector'")
        pytorch_versions = [
            v for v in all_versions
            if client.get_run(v.run_id).data.metrics.get("best_val_macro_f1", 0) > 0
        ]
        prev_best = max(
            (client.get_run(v.run_id).data.metrics.get("best_val_macro_f1", 0) for v in pytorch_versions),
            default=0.0,
        )

        if best_f1 >= prev_best:
            logger.info(f"New best model (f1={best_f1:.4f} >= prev_best={prev_best:.4f}) — registering")
            import tempfile
            import re as _re
            # Move model to CPU so the pyfunc can be loaded without CUDA
            model_cpu = model.cpu()
            tmp = tempfile.mkdtemp()
            model_path = Path(tmp) / "model.pth"
            torch.save(model_cpu, model_path)
            model.to(args.device)  # move back to GPU for any further use

            repo_root = str(Path(__file__).resolve().parents[1])
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=PADetectorPyfunc(),
                artifacts={"model_path": str(model_path)},
                registered_model_name="pa-detector",
                code_paths=[repo_root],
                pip_requirements=["torch==2.4.1", "transformers==4.44.2", "pandas==2.2.3"],
            )
            # Tag the new version as 'champion' — docker-compose always uses this alias
            all_v = client.search_model_versions("name='pa-detector'")
            new_ver = sorted(all_v, key=lambda x: int(x.version))[-1].version
            client.set_registered_model_alias("pa-detector", "champion", new_ver)
            logger.info(f"Alias 'champion' → pa-detector v{new_ver}")
            if Path("eval.json").exists():
                mlflow.log_artifact("eval.json")
        else:
            logger.info(f"Model (f1={best_f1:.4f}) did not beat prev_best={prev_best:.4f} — skipping registration")

        # Email notification for training run outcome
        try:
            from backend.app.services.notifier import notify_model_run
            best_pa_mae = val_metrics.get("pa_mae", 0.0) if Path(args.val_path).exists() else 0.0
            notify_model_run(
                run_name=args.run_name,
                f1=best_f1,
                pa_mae=best_pa_mae,
                registered=(best_f1 >= prev_best),
            )
        except Exception as _e:
            logger.debug(f"Email notification skipped: {_e}")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), models_dir / "checkpoint.pt")
    logger.info("checkpoint saved to models/checkpoint.pt")


if __name__ == "__main__":
    main()
