"""Pure cleaning functions — tested in isolation, then applied row-wise."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
WS_RE = re.compile(r"\s+")

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")


def clean_text(t: str) -> str:
    t = URL_RE.sub("<URL>", t)
    t = EMAIL_RE.sub("<EMAIL>", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def clean_dataframe(df: pd.DataFrame, min_len: int = 5) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df[df["text"].str.len() >= min_len]
    return df.reset_index(drop=True)


def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    for sub in RAW_DIR.iterdir():
        if not sub.is_dir():
            continue
        pq = sub / f"{sub.name}.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq)
        df_clean = clean_dataframe(df)
        out = INTERIM_DIR / f"{sub.name}.parquet"
        df_clean.to_parquet(out)


if __name__ == "__main__":
    main()
