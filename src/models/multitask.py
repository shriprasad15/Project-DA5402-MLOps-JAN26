from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import DistilBertConfig, DistilBertModel  # noqa: F401


@dataclass
class MultiTaskOutput:
    pa_logits: torch.Tensor
    sarcasm_logits: torch.Tensor
    tone_logits: torch.Tensor
    hidden: torch.Tensor


class PassiveAggressiveDetector(nn.Module):
    def __init__(
        self,
        pretrained: str = "distilbert-base-uncased",
        num_tone_classes: int = 5,
        dropout: float = 0.1,
        quantized: bool = False,
    ) -> None:
        super().__init__()
        self.quantized = quantized
        self.bert = DistilBertModel.from_pretrained(pretrained)
        self.dropout_layer = nn.Dropout(dropout)
        self.pa_head = nn.Linear(768, 1)
        self.sarcasm_head = nn.Linear(768, 1)
        self.tone_head = nn.Linear(768, num_tone_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> MultiTaskOutput:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        cls = hidden_states[:, 0, :]
        cls = self.dropout_layer(cls)
        return MultiTaskOutput(
            pa_logits=self.pa_head(cls),
            sarcasm_logits=self.sarcasm_head(cls),
            tone_logits=self.tone_head(cls),
            hidden=cls,
        )
