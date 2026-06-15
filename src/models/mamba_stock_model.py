from __future__ import annotations

import torch
from torch import nn


class MambaStockRegressor(nn.Module):
    """Stock-oriented selective state-space inspired block.

    This is a compact Mamba-style sequence model designed for tabular market features.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.in_proj = nn.Linear(input_size, hidden_size)
        self.select_gate = nn.Linear(input_size, hidden_size)
        self.state_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.in_proj(x))
        g = torch.sigmoid(self.select_gate(x))
        s = z * g
        s = self.state_conv(s.transpose(1, 2)).transpose(1, 2)
        s = self.norm(s)
        return self.head(s[:, -1, :]).squeeze(-1)
