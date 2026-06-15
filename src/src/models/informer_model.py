from __future__ import annotations

import torch
from torch import nn

from src.models.common_torch import PositionalEncoding


class InformerRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
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
        self.distill = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pos(self.embed(x))
        z = self.encoder(z)
        z = self.distill(z.transpose(1, 2)).transpose(1, 2)
        return self.head(z.mean(dim=1)).squeeze(-1)
