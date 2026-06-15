from __future__ import annotations

import torch
from torch import nn

from src.models.common_torch import PositionalEncoding


class AutoformerRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.mavg = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.embed = nn.Linear(input_size, hidden_size)
        self.pos = PositionalEncoding(hidden_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.trend_head = nn.Linear(input_size, 1)
        self.seasonal_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.mavg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        s = self.encoder(self.pos(self.embed(seasonal))).mean(dim=1)
        t = trend.mean(dim=1)
        return self.seasonal_head(s).squeeze(-1) + self.trend_head(t).squeeze(-1)
