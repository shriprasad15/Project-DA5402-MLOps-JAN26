from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# Maps string tone labels → integer class index
TONE_TO_IDX: dict[str, int] = {
    "neutral": 0,
    "friendly": 1,
    "assertive": 2,
    "aggressive": 3,
    "passive_aggressive": 4,
}


def get_tokenizer(model_name: str = "distilbert-base-uncased") -> PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(model_name)


def _tone_to_int(val: object) -> int:
    if isinstance(val, str):
        return TONE_TO_IDX.get(val.lower().strip(), 0)
    try:
        return int(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


class UnifiedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            str(row["text"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "pa_label": torch.tensor(
                float(row.get("passive_aggression", 0.0)), dtype=torch.float32
            ),
            "sarcasm_label": torch.tensor(float(row.get("sarcasm", 0.0)), dtype=torch.float32),
            "tone_label": torch.tensor(_tone_to_int(row.get("tone", 0)), dtype=torch.long),
        }
