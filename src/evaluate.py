from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score


def compute_metrics(df: pd.DataFrame) -> dict:
    tone_preds = df["tone_pred"].tolist()
    tone_labels = df["tone_label"].tolist()
    macro_f1 = float(f1_score(tone_labels, tone_preds, average="macro", zero_division=0))
    per_class_f1 = f1_score(
        tone_labels, tone_preds, average=None, labels=list(range(5)), zero_division=0
    ).tolist()
    pa_mae = float((df["pa_pred"] - df["pa_label"]).abs().mean())
    sarcasm_mae = float((df["sarcasm_pred"] - df["sarcasm_label"]).abs().mean())
    return {
        "macro_f1": macro_f1,
        "pa_mae": pa_mae,
        "sarcasm_mae": sarcasm_mae,
        "per_class_f1": per_class_f1,
    }


def generate_predictions(
    checkpoint: Path,
    test_path: Path,
    out_path: Path,
    device: str = "cpu",
) -> None:
    from src.features.tokenize import UnifiedDataset, get_tokenizer
    from torch.utils.data import DataLoader

    tokenizer = get_tokenizer()
    df = pd.read_parquet(test_path)
    ds = UnifiedDataset(df, tokenizer)
    loader = DataLoader(ds, batch_size=32)

    from src.models.multitask import PassiveAggressiveDetector
    model = PassiveAggressiveDetector()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    pa_preds, sa_preds, tone_preds = [], [], []
    pa_labels, sa_labels, tone_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model(ids, mask)
            pa_preds.extend(out.pa_logits.squeeze().cpu().tolist())
            sa_preds.extend(out.sarcasm_logits.squeeze().cpu().tolist())
            tone_preds.extend(out.tone_logits.argmax(-1).cpu().tolist())
            pa_labels.extend(batch["pa_label"].tolist())
            sa_labels.extend(batch["sarcasm_label"].tolist())
            tone_labels.extend(batch["tone_label"].tolist())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "pa_pred": pa_preds, "pa_label": pa_labels,
        "sarcasm_pred": sa_preds, "sarcasm_label": sa_labels,
        "tone_pred": tone_preds, "tone_label": tone_labels,
    }).to_parquet(out_path, index=False)


def main() -> None:
    pred_path = Path("models/predictions.parquet")
    checkpoint = Path("models/checkpoint.pt")
    test_path = Path("data/processed/test.parquet")

    if not pred_path.exists():
        if not checkpoint.exists():
            raise FileNotFoundError("models/checkpoint.pt not found — run train first")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generate_predictions(checkpoint, test_path, pred_path, device=device)

    df = pd.read_parquet(pred_path)
    metrics = compute_metrics(df)
    Path("eval.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
