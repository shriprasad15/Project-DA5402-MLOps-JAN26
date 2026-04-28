from __future__ import annotations

import pytest
import torch

from src.models.loss import UncertaintyWeightedLoss


@pytest.mark.unit
def test_loss_positive() -> None:
    criterion = UncertaintyWeightedLoss(n_tasks=3)
    losses = [torch.tensor(0.5), torch.tensor(0.3), torch.tensor(0.7)]
    result = criterion(losses)
    assert result.item() > 0


@pytest.mark.unit
def test_no_nans() -> None:
    criterion = UncertaintyWeightedLoss(n_tasks=3)
    losses = [torch.tensor(0.5), torch.tensor(0.3), torch.tensor(0.7)]
    result = criterion(losses)
    assert not torch.isnan(result)
