"""Source adapters — one class per upstream dataset. Each returns a
DataFrame with at least ['text', 'source'] columns.

Every adapter caches raw downloads locally so `dvc repro ingest` is
deterministic and network-independent after the first run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class SourceAdapter(ABC):
    name: str

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_file(self) -> Path:
        return self.cache_dir / f"{self.name}.parquet"

    def load(self, limit: int | None = None) -> pd.DataFrame:
        if self.cache_file.exists():
            df = pd.read_parquet(self.cache_file)
        else:
            df = self._download()
            df["source"] = self.name
            df.to_parquet(self.cache_file)
        return df.head(limit) if limit else df

    @abstractmethod
    def _download(self) -> pd.DataFrame: ...


class SarcasmHeadlinesAdapter(SourceAdapter):
    name = "sarcasm_headlines"
    URL = "https://raw.githubusercontent.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/master/Sarcasm_Headlines_Dataset.json"

    def _download(self) -> pd.DataFrame:
        df = pd.read_json(self.URL, lines=True)
        df = df.rename(columns={"headline": "text", "is_sarcastic": "sarcasm"})
        df["sarcasm"] = df["sarcasm"].astype(float)
        return df[["text", "sarcasm"]]


class GoEmotionsAdapter(SourceAdapter):
    name = "goemotions"

    def _download(self) -> pd.DataFrame:
        from datasets import load_dataset

        ds = load_dataset("go_emotions", "simplified", split="train")
        df = ds.to_pandas()
        df = df.rename(columns={"text": "text"})
        return df[["text", "labels"]]


class ISarcasmAdapter(SourceAdapter):
    name = "isarcasm"

    def _download(self) -> pd.DataFrame:
        from datasets import load_dataset

        ds = load_dataset(
            "jkhedri/psychology-dataset", split="train"
        )  # placeholder — swap for iSarcasm mirror
        df = ds.to_pandas().rename(columns={"question": "text"})
        df["sarcasm"] = 0.8
        return df[["text", "sarcasm"]].head(5000)


class EnronSubsetAdapter(SourceAdapter):
    name = "enron_subset"

    def _download(self) -> pd.DataFrame:
        from datasets import load_dataset

        ds = load_dataset("snoop2head/enron_aeslc_emails", split="train")
        df = ds.to_pandas().rename(columns={"email_body": "text"})
        return df[["text"]].head(20000)


ALL_ADAPTERS: list[type[SourceAdapter]] = [
    SarcasmHeadlinesAdapter,
    GoEmotionsAdapter,
    ISarcasmAdapter,
    EnronSubsetAdapter,
]
