from __future__ import annotations

import torch
from torch import nn


class NBEATSRegressor(nn.Module):
    def __init__(self, input_size: int, seq_len: int, hidden_size: int, n_blocks: int = 3):
        super().__init__()
        flat_size = input_size * seq_len
        blocks = []
        for _ in range(n_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Linear(flat_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, flat_size),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(flat_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        residual = x.reshape(b, -1)
        for block in self.blocks:
            backcast = block(residual)
            residual = residual - backcast
        return self.head(residual).squeeze(-1)
