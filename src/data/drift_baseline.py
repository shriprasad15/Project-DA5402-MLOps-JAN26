"""Compute reference statistics for drift detection."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed/train.parquet")
REF_DIR = Path("data/reference")


def compute_baseline(df: pd.DataFrame) -> dict:
    lengths = df["text"].str.len().to_numpy()
    tokens = [w for line in df["text"] for w in line.lower().split()]
    vocab = dict(Counter(tokens).most_common(10000))
    return {
        "length_mean": float(np.mean(lengths)),
        "length_std": float(np.std(lengths)),
        "length_quantiles": {
            str(q): float(np.quantile(lengths, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        },
        "vocab": vocab,
    }


def main() -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED)
    stats = compute_baseline(df)
    (REF_DIR / "feature_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
