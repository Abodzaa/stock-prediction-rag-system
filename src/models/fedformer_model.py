from __future__ import annotations

import torch
from torch import nn

from src.models.common_torch import PositionalEncoding


class FEDformerRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_heads: int = 4, n_layers: int = 2, keep_freq: int = 16):
        super().__init__()
        self.keep_freq = keep_freq
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
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = torch.fft.rfft(x, dim=1)
        if xf.shape[1] > self.keep_freq:
            xf[:, self.keep_freq :, :] = 0
        x_rec = torch.fft.irfft(xf, n=x.shape[1], dim=1)
        z = self.encoder(self.pos(self.embed(x_rec)))
        return self.head(z.mean(dim=1)).squeeze(-1)
