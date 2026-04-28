from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutput

from src.models.multitask import MultiTaskOutput, PassiveAggressiveDetector


def _make_mock_bert() -> MagicMock:
    mock_bert = MagicMock()
    mock_output = BaseModelOutput(last_hidden_state=torch.zeros(2, 16, 768))
    mock_bert.return_value = mock_output
    return mock_bert


@pytest.mark.unit
def test_forward_shapes() -> None:
    with patch(
        "src.models.multitask.DistilBertModel.from_pretrained", return_value=_make_mock_bert()
    ):
        model = PassiveAggressiveDetector(pretrained="distilbert-base-uncased")
    input_ids = torch.zeros(2, 16, dtype=torch.long)
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    out = model(input_ids, attention_mask)
    assert isinstance(out, MultiTaskOutput)
    assert out.pa_logits.shape == (2, 1)
    assert out.sarcasm_logits.shape == (2, 1)
    assert out.tone_logits.shape == (2, 5)
    assert out.hidden.shape == (2, 768)


@pytest.mark.unit
def test_quantized_default_false() -> None:
    with patch(
        "src.models.multitask.DistilBertModel.from_pretrained", return_value=_make_mock_bert()
    ):
        model = PassiveAggressiveDetector()
    assert model.quantized is False
