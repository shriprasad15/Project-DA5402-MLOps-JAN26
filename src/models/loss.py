from __future__ import annotations

import torch
import torch.nn as nn


class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, n_tasks: int = 3) -> None:
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        total = torch.zeros(1, device=self.log_sigma.device)
        for i, loss in enumerate(losses):
            total = total + (loss / (2 * torch.exp(2 * self.log_sigma[i]))) + self.log_sigma[i]
        return total.squeeze()
