from __future__ import annotations

import torch
from torch import nn

from src.models.common_torch import PositionalEncoding


class PatchTSTRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, patch_len: int = 8, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.patch_len = patch_len
        self.patch_proj = nn.Linear(input_size * patch_len, hidden_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pos = PositionalEncoding(hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        if t < self.patch_len:
            pad = self.patch_len - t
            x = torch.cat([x[:, :1, :].repeat(1, pad, 1), x], dim=1)
            t = x.shape[1]

        n_patches = t // self.patch_len
        x = x[:, -n_patches * self.patch_len :, :]
        x = x.reshape(b, n_patches, self.patch_len * f)
        tokens = self.patch_proj(x)
        tokens = self.pos(tokens)
        enc = self.encoder(tokens)
        pooled = enc.mean(dim=1)
        return self.head(pooled).squeeze(-1)
