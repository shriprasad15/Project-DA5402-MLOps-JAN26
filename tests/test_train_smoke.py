from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from transformers.modeling_outputs import BaseModelOutput

from src.features.tokenize import UnifiedDataset
from src.train import train_one_epoch


def _make_mock_bert() -> MagicMock:
    mock_bert = MagicMock()

    def _forward(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        seq_len = kwargs["input_ids"].shape[1]
        return BaseModelOutput(last_hidden_state=torch.zeros(batch_size, seq_len, 768))

    mock_bert.side_effect = _forward
    return mock_bert


@pytest.mark.slow
def test_train_one_epoch_smoke() -> None:
    df = pd.DataFrame(
        {
            "text": ["hello world"] * 5,
            "passive_aggression": [0.0, 1.0, 0.5, 0.0, 1.0],
            "sarcasm": [0.0, 0.0, 1.0, 1.0, 0.5],
            "tone": [0, 1, 2, 3, 4],
        }
    )

    mock_tokenizer = MagicMock()
    mock_tokenizer.side_effect = lambda text, **kw: {
        "input_ids": torch.zeros(1, kw.get("max_length", 16), dtype=torch.long),
        "attention_mask": torch.ones(1, kw.get("max_length", 16), dtype=torch.long),
    }

    with patch(
        "src.models.multitask.DistilBertModel.from_pretrained", return_value=_make_mock_bert()
    ):
        from src.models.multitask import PassiveAggressiveDetector

        model = PassiveAggressiveDetector()

    dataset = UnifiedDataset(df, mock_tokenizer, max_length=16)
    loader = DataLoader(dataset, batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    loss = train_one_epoch(model, loader, optimizer, "cpu", max_steps=2)
    assert loss > 0
