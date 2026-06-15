from __future__ import annotations

import torch
from torch import nn


class TFTRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_heads: int = 4):
        super().__init__()
        self.var_select = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Softmax(dim=-1),
        )
        self.enc = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=n_heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.var_select(x)
        xw = x * w
        h, _ = self.enc(xw)
        a, _ = self.attn(h, h, h)
        g = self.gate(a)
        z = g * a + (1 - g) * h
        return self.head(z[:, -1, :]).squeeze(-1)
