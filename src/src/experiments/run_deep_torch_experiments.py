#!/usr/bin/env python3
"""Walk-forward deep-model experiments with hyperparameter tuning.

Implements the required model list from the research plan:
- Primary: TFT, PatchTST, Mamba, Informer, Autoformer, FEDformer
- Baselines: LSTM, GRU, NBEATS, NHITS
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.model_factory import build_torch_model


PRIMARY_MODELS = ["TFT", "PatchTST", "MambaStock", "Informer", "Autoformer", "FEDformer"]
BASELINE_MODELS = ["LSTM", "GRU", "NBEATS", "NHITS"]
ALL_MODELS = PRIMARY_MODELS + BASELINE_MODELS
MODEL_ALIASES = {"Mamba": "MambaStock"}


DEFAULT_FEATURES_CSV = Path("data/processed/features/research_features_daily.csv")
DEFAULT_RESULTS_CSV = Path("reports/deep_torch_model_results.csv")
DEFAULT_SUMMARY_JSON = Path("reports/deep_torch_model_summary.json")
DEFAULT_SUMMARY_MD = Path("reports/deep_torch_model_summary.md")
DEFAULT_ARTIFACT_DIR = Path("artifacts/models/deep_torch")


@dataclass(frozen=True)
class TrialSpec:
    model_name: str
    feature_group: str
    lag_window: int
    hidden_size: int
    lr: float
    epochs: int
    patience: int
    batch_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep PyTorch model sweep runner.")
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--target", default="target_h1", choices=["target_h1", "target_h5"])
    parser.add_argument("--models", default=",".join(ALL_MODELS))
    parser.add_argument("--sentiment-prefix", default="g5_sent_")
    parser.add_argument("--feature-groups", default="", help="Optional CSV of feature group names to run.")

    parser.add_argument("--lag-window", type=int, default=64)
    parser.add_argument("--lag-window-grid", default="32,64,128")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--hidden-size-grid", default="32,64")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-grid", default="0.001,0.0005")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--epochs-grid", default="20,50,100")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--patience-grid", default="5,10,20")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-size-grid", default="64")

    parser.add_argument("--tune", action="store_true")

    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=int, default=252)
    parser.add_argument("--min-train-size", type=int, default=1260)
    parser.add_argument("--embargo", type=int, default=5)
    parser.add_argument("--val-size", type=int, default=252)

    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def parse_csv_items(raw: str) -> list[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one CSV item")
    return vals


def parse_int_grid(raw: str) -> list[int]:
    vals = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not vals:
        raise ValueError("Expected at least one integer in grid")
    return vals


def parse_float_grid(raw: str) -> list[float]:
    vals = sorted({float(x.strip()) for x in raw.split(",") if x.strip()})
    if not vals:
        raise ValueError("Expected at least one float in grid")
    return vals


def pick_device(raw: str) -> torch.device:
    if raw == "cpu":
        return torch.device("cpu")
    if raw == "mps":
        return torch.device("mps")
    if raw == "cuda":
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    pred_sign = np.sign(y_pred)
    true_sign = np.sign(y_true)
    directional_accuracy = float(np.mean(pred_sign == true_sign))

    strategy_ret = pred_sign * y_true
    std = float(np.std(strategy_ret, ddof=1)) if len(strategy_ret) > 1 else np.nan
    if np.isfinite(std) and std > 0:
        sharpe = float(np.sqrt(252) * np.mean(strategy_ret) / std)
    else:
        sharpe = float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": directional_accuracy,
        "sharpe": sharpe,
    }


def walkforward_splits(n: int, n_splits: int, test_size: int, min_train_size: int, embargo: int):
    required = n_splits * test_size
    if n <= required + min_train_size + embargo:
        raise ValueError(
            f"Not enough rows ({n}) for walk-forward setup "
            f"(splits={n_splits}, test_size={test_size}, min_train={min_train_size}, embargo={embargo})."
        )

    start = n - required
    for i in range(n_splits):
        test_start = start + i * test_size
        test_end = min(test_start + test_size, n)
        train_end = test_start - embargo
        if train_end < min_train_size:
            continue
        yield i + 1, 0, train_end, test_start, test_end


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lag_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lag_window <= 0:
        raise ValueError("lag_window must be positive")

    work = df[["Date", target_col] + feature_cols].copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.dropna(subset=["Date", target_col]).sort_values("Date").reset_index(drop=True)

    values = work[feature_cols].to_numpy(dtype=float)
    y = work[target_col].to_numpy(dtype=float)
    ds = work["Date"].to_numpy()

    xs, ys, ts = [], [], []
    for i in range(lag_window, len(work)):
        xs.append(values[i - lag_window : i])
        ys.append(y[i])
        ts.append(ds[i])

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(ts)


def standardize_by_train(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Feature-wise normalization across all train timesteps.
    mean = np.nanmean(x_train.reshape(-1, x_train.shape[-1]), axis=0)
    std = np.nanstd(x_train.reshape(-1, x_train.shape[-1]), axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    def tr(x: np.ndarray) -> np.ndarray:
        z = (x - mean[None, None, :]) / std[None, None, :]
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return z

    return tr(x_train), tr(x_val), tr(x_test)


def train_one_fold(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    epochs: int,
    patience: int,
    batch_size: int,
    hidden_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, float, dict, dict[str, torch.Tensor] | None]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_torch_model(
        model_name=model_name,
        input_size=x_train.shape[-1],
        seq_len=x_train.shape[1],
        hidden_size=hidden_size,
    ).to(device)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    test_x = torch.from_numpy(x_test).to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    trained_epochs = 0

    for epoch in range(1, epochs + 1):
        trained_epochs = epoch
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                vloss = criterion(pred, yb).item()
                val_losses.append(vloss)

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred = model(test_x).detach().cpu().numpy()

    train_meta = {
        "epochs_trained": trained_epochs,
        "best_val_loss": best_val,
    }
    return y_pred, best_val, train_meta, best_state


def evaluate_trial(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    spec: TrialSpec,
    n_splits: int,
    test_size: int,
    min_train_size: int,
    embargo: int,
    val_size: int,
    device: torch.device,
    seed: int,
) -> dict:
    x_all, y_all, _ = build_sequences(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        lag_window=spec.lag_window,
    )

    fold_rows = []
    y_true_parts = []
    y_pred_parts = []
    epoch_parts = []
    checkpoint_state = None

    for fold_id, train_start, train_end, test_start, test_end in walkforward_splits(
        n=len(x_all),
        n_splits=n_splits,
        test_size=test_size,
        min_train_size=min_train_size,
        embargo=embargo,
    ):
        tr_x = x_all[train_start:train_end]
        tr_y = y_all[train_start:train_end]

        if len(tr_x) <= val_size + 32:
            raise RuntimeError("Training window too small for requested val_size.")

        split = len(tr_x) - val_size
        x_train, y_train = tr_x[:split], tr_y[:split]
        x_val, y_val = tr_x[split:], tr_y[split:]
        x_test, y_test = x_all[test_start:test_end], y_all[test_start:test_end]

        x_train, x_val, x_test = standardize_by_train(x_train=x_train, x_val=x_val, x_test=x_test)

        y_pred, _, meta, best_state = train_one_fold(
            model_name=spec.model_name,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            epochs=spec.epochs,
            patience=spec.patience,
            batch_size=spec.batch_size,
            hidden_size=spec.hidden_size,
            lr=spec.lr,
            device=device,
            seed=seed + fold_id,
        )

        if best_state is not None:
            checkpoint_state = best_state

        m = compute_metrics(y_true=y_test, y_pred=y_pred)
        fold_rows.append({"fold": fold_id, **m, "n_test": int(len(y_test)), **meta})
        y_true_parts.append(y_test)
        y_pred_parts.append(y_pred)
        epoch_parts.append(meta["epochs_trained"])

    if not fold_rows:
        raise RuntimeError("No valid folds generated for trial")

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    overall = compute_metrics(y_true=y_true, y_pred=y_pred)

    return {
        "overall": overall,
        "n_rows": int(len(x_all)),
        "n_features": int(x_all.shape[-1]),
        "seq_len": int(x_all.shape[1]),
        "n_test_total": int(len(y_true)),
        "avg_epochs_trained": float(np.mean(epoch_parts)),
        "fold_metrics": fold_rows,
        "checkpoint_state": checkpoint_state,
    }


def build_trials(
    models: list[str],
    feature_groups: dict[str, list[str]],
    lag_windows: list[int],
    hidden_sizes: list[int],
    lrs: list[float],
    epochs_grid: list[int],
    patience_grid: list[int],
    batch_sizes: list[int],
) -> list[TrialSpec]:
    trials = []
    for model_name, feature_group, lag, hs, lr, ep, pa, bs in product(
        models,
        feature_groups.keys(),
        lag_windows,
        hidden_sizes,
        lrs,
        epochs_grid,
        patience_grid,
        batch_sizes,
    ):
        trials.append(
            TrialSpec(
                model_name=model_name,
                feature_group=feature_group,
                lag_window=lag,
                hidden_size=hs,
                lr=lr,
                epochs=ep,
                patience=pa,
                batch_size=bs,
            )
        )
    return trials


def write_summary_md(summary_md: Path, results: pd.DataFrame) -> None:
    summary_md.parent.mkdir(parents=True, exist_ok=True)

    ok = results[results["status"] == "ok"].copy()
    prim = ok[ok["model_name"].isin(PRIMARY_MODELS)]
    base = ok[ok["model_name"].isin(BASELINE_MODELS)]
    bad = results[results["status"] != "ok"]

    lines = [
        "# Deep Torch Experiment Summary",
        "",
        "## Primary Models (Top 15 by RMSE)",
    ]

    if len(prim) == 0:
        lines.append("- none")
    else:
        for _, r in prim.sort_values("rmse").head(15).iterrows():
            lines.append(
                f"- {r['model_name']} | {r['feature_group']} | lag={int(r['lag_window'])} | "
                f"hidden={int(r['hidden_size'])} | lr={r['lr']:.5f} | ep={int(r['epochs'])} | "
                f"pat={int(r['patience'])}: rmse={r['rmse']:.6f}, dir_acc={r['directional_accuracy']:.4f}, sharpe={r['sharpe']:.4f}"
            )

    lines.extend(["", "## Baselines (Top 15 by RMSE)"])
    if len(base) == 0:
        lines.append("- none")
    else:
        for _, r in base.sort_values("rmse").head(15).iterrows():
            lines.append(
                f"- {r['model_name']} | {r['feature_group']} | lag={int(r['lag_window'])} | "
                f"hidden={int(r['hidden_size'])} | lr={r['lr']:.5f} | ep={int(r['epochs'])} | "
                f"pat={int(r['patience'])}: rmse={r['rmse']:.6f}, dir_acc={r['directional_accuracy']:.4f}, sharpe={r['sharpe']:.4f}"
            )

    lines.extend(["", "## Failed / Skipped"])
    if len(bad) == 0:
        lines.append("- none")
    else:
        for _, r in bad.head(20).iterrows():
            lines.append(f"- {r['model_name']} | {r['feature_group']} | {r['status']} | {r['note']}")

    lines.append("")
    summary_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {args.features_csv}")

    models = [MODEL_ALIASES.get(m, m) for m in parse_csv_items(args.models)]
    # Keep order while dropping duplicates introduced by aliases.
    models = list(dict.fromkeys(models))
    unknown = sorted(set(models).difference(ALL_MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")

    if args.tune:
        lag_windows = parse_int_grid(args.lag_window_grid)
        hidden_sizes = parse_int_grid(args.hidden_size_grid)
        lrs = parse_float_grid(args.lr_grid)
        epochs_grid = parse_int_grid(args.epochs_grid)
        patience_grid = parse_int_grid(args.patience_grid)
        batch_sizes = parse_int_grid(args.batch_size_grid)
    else:
        lag_windows = [args.lag_window]
        hidden_sizes = [args.hidden_size]
        lrs = [args.lr]
        epochs_grid = [args.epochs]
        patience_grid = [args.patience]
        batch_sizes = [args.batch_size]

    device = pick_device(args.device)

    df = pd.read_csv(args.features_csv)
    g1 = [c for c in df.columns if c.startswith("g1_")]
    g2 = [c for c in df.columns if c.startswith("g2_")]
    g3 = [c for c in df.columns if c.startswith("g3_")]
    g5 = [c for c in df.columns if c.startswith(args.sentiment_prefix)]

    feature_groups = {
        "G1_price_only": g1,
        "G2_price_technical": g1 + g2,
        "G3_plus_breadth": g1 + g2 + g3,
    }
    if g5:
        feature_groups["G4_price_plus_sentiment"] = g1 + g2 + g3 + g5
        feature_groups["G5_sentiment_only"] = g5

    if args.feature_groups.strip():
        requested_groups = parse_csv_items(args.feature_groups)
        unknown_groups = sorted(set(requested_groups).difference(feature_groups.keys()))
        if unknown_groups:
            raise ValueError(f"Unknown feature groups requested: {unknown_groups}")
        feature_groups = {name: feature_groups[name] for name in requested_groups}

    trials = build_trials(
        models=models,
        feature_groups=feature_groups,
        lag_windows=lag_windows,
        hidden_sizes=hidden_sizes,
        lrs=lrs,
        epochs_grid=epochs_grid,
        patience_grid=patience_grid,
        batch_sizes=batch_sizes,
    )

    rows = []
    best_key_to_result = {}
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = args.artifact_dir / args.target
    ckpt_root.mkdir(parents=True, exist_ok=True)

    for i, spec in enumerate(trials, start=1):
        try:
            result = evaluate_trial(
                df=df,
                feature_cols=feature_groups[spec.feature_group],
                target_col=args.target,
                spec=spec,
                n_splits=args.n_splits,
                test_size=args.test_size,
                min_train_size=args.min_train_size,
                embargo=args.embargo,
                val_size=args.val_size,
                device=device,
                seed=args.seed,
            )

            record = {
                **asdict(spec),
                "target": args.target,
                "status": "ok",
                "note": "",
                **result["overall"],
                "n_rows": result["n_rows"],
                "n_features": result["n_features"],
                "seq_len": result["seq_len"],
                "n_test_total": result["n_test_total"],
                "avg_epochs_trained": result["avg_epochs_trained"],
            }

            artifact_name = (
                f"{args.target}_{spec.feature_group}_{spec.model_name}_"
                f"lag{spec.lag_window}_h{spec.hidden_size}_"
                f"lr{spec.lr}_e{spec.epochs}_p{spec.patience}.pt"
            ).replace("/", "_")
            artifact_path = ckpt_root / artifact_name
            torch.save(
                {
                    "model_name": spec.model_name,
                    "feature_group": spec.feature_group,
                    "target": args.target,
                    "config": asdict(spec),
                    "metrics": result["overall"],
                    "n_features": int(result["n_features"]),
                    "seq_len": int(result["seq_len"]),
                    "state_dict": result["checkpoint_state"],
                },
                artifact_path,
            )
            record["artifact_path"] = str(artifact_path)

            # Keep best config per (model, feature_group) by RMSE.
            key = (spec.model_name, spec.feature_group)
            if key not in best_key_to_result or record["rmse"] < best_key_to_result[key]["rmse"]:
                best_key_to_result[key] = record

            rows.append(record)
        except Exception as exc:
            rows.append(
                {
                    **asdict(spec),
                    "target": args.target,
                    "status": "error",
                    "note": str(exc),
                    "rmse": np.nan,
                    "mae": np.nan,
                    "directional_accuracy": np.nan,
                    "sharpe": np.nan,
                    "n_rows": 0,
                    "n_features": len(feature_groups[spec.feature_group]),
                    "seq_len": spec.lag_window,
                    "n_test_total": 0,
                    "avg_epochs_trained": 0,
                    "artifact_path": "",
                }
            )

        if i % 5 == 0:
            print(f"Progress: {i}/{len(trials)} trials complete")

    results = pd.DataFrame(rows)
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_csv, index=False)

    # Save best model configs table and best model placeholders for reproducibility.
    best_df = pd.DataFrame(best_key_to_result.values())
    if len(best_df) > 0:
        best_df = best_df.sort_values("rmse")
        best_cfg_path = args.results_csv.with_name(args.results_csv.stem + "_best_configs.csv")
        best_df.to_csv(best_cfg_path, index=False)
    else:
        best_cfg_path = args.results_csv.with_name(args.results_csv.stem + "_best_configs.csv")
        pd.DataFrame(columns=["model_name", "feature_group", "rmse"]).to_csv(best_cfg_path, index=False)

    # Persist metadata for reproduction.
    artifact_meta = {
        "target": args.target,
        "device": str(device),
        "models": models,
        "lag_windows": lag_windows,
        "hidden_sizes": hidden_sizes,
        "learning_rates": lrs,
        "epochs_grid": epochs_grid,
        "patience_grid": patience_grid,
        "batch_sizes": batch_sizes,
        "n_trials": int(len(results)),
        "ok_trials": int((results["status"] == "ok").sum()),
        "error_trials": int((results["status"] == "error").sum()),
        "best_configs_csv": str(best_cfg_path),
        "checkpoint_root": str(ckpt_root),
    }

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.artifact_dir / f"{args.target}_deep_torch_sweep_metadata.json"
    meta_path.write_text(json.dumps(artifact_meta, indent=2), encoding="utf-8")

    summary_payload = {
        "target": args.target,
        "device": str(device),
        "trial_counts": {
            "total": int(len(results)),
            "ok": int((results["status"] == "ok").sum()),
            "error": int((results["status"] == "error").sum()),
        },
        "best_overall": results[results["status"] == "ok"].sort_values("rmse").head(20).to_dict(orient="records"),
        "best_primary": results[(results["status"] == "ok") & (results["model_name"].isin(PRIMARY_MODELS))]
        .sort_values("rmse")
        .head(20)
        .to_dict(orient="records"),
        "best_baseline": results[(results["status"] == "ok") & (results["model_name"].isin(BASELINE_MODELS))]
        .sort_values("rmse")
        .head(20)
        .to_dict(orient="records"),
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    write_summary_md(summary_md=args.summary_md, results=results)

    print("Deep torch sweep complete.")
    print(f"Target: {args.target}")
    print(f"Trials: {len(results)} | ok={(results['status'] == 'ok').sum()} | error={(results['status'] == 'error').sum()}")
    print(f"Results CSV: {args.results_csv}")
    print(f"Best configs CSV: {best_cfg_path}")
    print(f"Summary JSON: {args.summary_json}")
    print(f"Summary MD: {args.summary_md}")
    print(f"Artifact metadata: {meta_path}")


if __name__ == "__main__":
    main()
