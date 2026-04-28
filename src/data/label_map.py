"""Map every source's native labels onto the unified schema.

Unified schema:
    text, passive_aggression [0-1], sarcasm [0-1],
    tone {neutral, friendly, assertive, aggressive, passive_aggressive},
    source, weak_label (bool)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from contracts.tone_enum import Tone

UNIFIED_COLS = ["text", "passive_aggression", "sarcasm", "tone", "source", "weak_label"]

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")

# GoEmotions simplified label indices we care about
GOEMOTIONS_ANGER = {2, 3, 16}  # anger, annoyance, disapproval
GOEMOTIONS_JOY = {17, 18, 0}  # joy, love, admiration
GOEMOTIONS_NEUTRAL = {27}


def _goemotions_tone(labels: list[int]) -> str:
    if any(lbl in GOEMOTIONS_ANGER for lbl in labels):
        return Tone.AGGRESSIVE.value if 2 in labels else Tone.ASSERTIVE.value
    if any(lbl in GOEMOTIONS_JOY for lbl in labels):
        return Tone.FRIENDLY.value
    return Tone.NEUTRAL.value


def to_unified(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["text"] = df["text"]
    out["passive_aggression"] = 0.0
    out["sarcasm"] = 0.0
    out["tone"] = Tone.NEUTRAL.value
    out["source"] = source
    out["weak_label"] = True

    if source in {"sarcasm_headlines", "isarcasm"}:
        out["sarcasm"] = df["sarcasm"].astype(float)
        out.loc[out["sarcasm"] > 0.5, "tone"] = Tone.PASSIVE_AGGRESSIVE.value
        out.loc[out["sarcasm"] > 0.5, "passive_aggression"] = 0.75
        out.loc[(out["sarcasm"] >= 0.3) & (out["sarcasm"] <= 0.5), "passive_aggression"] = 0.3
    elif source == "goemotions":
        out["tone"] = df["labels"].map(_goemotions_tone)
    elif source == "enron_subset":
        # Enron is unlabeled email corpus — used for text-distribution grounding;
        # keep as neutral weak label, heuristic PA scoring applied post-synthesis.
        pass
    elif source.startswith("synthetic_"):
        for c in ["passive_aggression", "sarcasm", "tone"]:
            out[c] = df[c]

    return out[UNIFIED_COLS]


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for pq in INTERIM_DIR.glob("*.parquet"):
        df = pd.read_parquet(pq)
        frames.append(to_unified(df, source=pq.stem))
    combined = pd.concat(frames, ignore_index=True)
    # split 80/10/10
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(combined)
    combined.iloc[: int(0.8 * n)].to_parquet(PROCESSED_DIR / "train.parquet")
    combined.iloc[int(0.8 * n) : int(0.9 * n)].to_parquet(PROCESSED_DIR / "val.parquet")
    combined.iloc[int(0.9 * n) :].to_parquet(PROCESSED_DIR / "test.parquet")


if __name__ == "__main__":
    main()
