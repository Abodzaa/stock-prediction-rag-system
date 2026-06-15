#!/usr/bin/env python3
"""Model registry and hyperparameter grid definitions for deep time-series sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product


PRIMARY_MODELS = [
    "TFT",
    "PatchTST",
    "MambaStock",
    "Informer",
    "Autoformer",
    "FEDformer",
]

BASELINE_MODELS = [
    "LSTM",
    "GRU",
    "NBEATS",
    "NHITS",
]

ALL_MODELS = PRIMARY_MODELS + BASELINE_MODELS + ["Mamba"]


@dataclass(frozen=True)
class DeepRunSpec:
    model_name: str
    feature_group: str
    input_size: int
    max_steps: int
    patience: int
    hidden_size: int
    batch_size: int


def build_hyperparameter_grid(
    model_name: str,
    feature_group: str,
    input_sizes: list[int],
    max_steps_grid: list[int],
    patience_grid: list[int],
) -> list[DeepRunSpec]:
    """Build model-specific hyperparameter candidates.

    We keep grid sizes modest by design to avoid p-hacking and excessive compute.
    """
    if model_name not in ALL_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name in {"TFT", "PatchTST", "Informer", "Autoformer", "FEDformer", "Mamba", "MambaStock"}:
        hidden_sizes = [32, 64]
        batch_sizes = [32]
    else:
        hidden_sizes = [32, 64, 128]
        batch_sizes = [32, 64]

    specs = []
    for input_size, max_steps, patience, hidden_size, batch_size in product(
        input_sizes, max_steps_grid, patience_grid, hidden_sizes, batch_sizes
    ):
        specs.append(
            DeepRunSpec(
                model_name=model_name,
                feature_group=feature_group,
                input_size=input_size,
                max_steps=max_steps,
                patience=patience,
                hidden_size=hidden_size,
                batch_size=batch_size,
            )
        )
    return specs
