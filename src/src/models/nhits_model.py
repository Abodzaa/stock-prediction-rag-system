from __future__ import annotations

import torch
from torch import nn


class NHITSRegressor(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.proj = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        p2 = self.pool2(xt).flatten(1)
        p4 = self.pool4(xt).flatten(1)
        raw = xt.flatten(1)
        feats = torch.cat([raw, p2, p4], dim=1)
        return self.proj(feats).squeeze(-1)
