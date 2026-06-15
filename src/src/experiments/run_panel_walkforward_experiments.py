#!/usr/bin/env python3
"""Run panel walk-forward experiments on (Date, Symbol) feature tables."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from joblib import dump
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


DEFAULT_FEATURES_CSV = Path("data/processed/features/research_features_panel.csv")
DEFAULT_GROUP_RESULTS = Path("reports/panel_feature_group_results.csv")
DEFAULT_MODEL_RESULTS = Path("reports/panel_model_set_results.csv")
DEFAULT_MODEL_FAMILY_RESULTS = Path("reports/panel_model_family_results.csv")
DEFAULT_SUMMARY_MD = Path("reports/panel_experiment_walkforward_summary.md")
DEFAULT_SUMMARY_JSON = Path("reports/panel_experiment_walkforward_summary.json")
DEFAULT_MODEL_ARTIFACTS_DIR = Path("artifacts/models/panel")

DEFAULT_MODEL_FAMILIES = ["elastic_net", "ridge", "random_forest", "hist_gbr", "mlp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run panel walk-forward experiment suite.")
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--group-results", type=Path, default=DEFAULT_GROUP_RESULTS)
    parser.add_argument("--model-results", type=Path, default=DEFAULT_MODEL_RESULTS)
    parser.add_argument("--model-family-results", type=Path, default=DEFAULT_MODEL_FAMILY_RESULTS)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--model-artifacts-dir", type=Path, default=DEFAULT_MODEL_ARTIFACTS_DIR)
    parser.add_argument("--target", default="target_h1", choices=["target_h1", "target_h5"])

    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=int, default=126, help="Test size in days per split.")
    parser.add_argument("--min-train-size", type=int, default=1260, help="Minimum train size in days.")
    parser.add_argument("--embargo", type=int, default=5, help="Embargo in days.")

    parser.add_argument("--models", default=",".join(DEFAULT_MODEL_FAMILIES))
    parser.add_argument("--lag-window", type=int, default=0)
    parser.add_argument("--lag-window-grid", default="0,5,10")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--epochs-grid", default="150,300")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--patience-grid", default="10,20")
    parser.add_argument("--sentiment-prefix", default="g5_sent_")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=350000)
    parser.add_argument("--max-test-rows", type=int, default=150000)
    parser.add_argument("--tune", action="store_true")
    return parser.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected comma-separated values.")
    return vals


def parse_int_grid(raw: str) -> list[int]:
    vals = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not vals:
        raise ValueError("Expected integer grid values.")
    return vals


def walkforward_date_splits(
    unique_dates: pd.DatetimeIndex,
    n_splits: int,
    test_size: int,
    min_train_size: int,
    embargo: int,
):
    n = len(unique_dates)
    required = n_splits * test_size
    if n <= required + min_train_size + embargo:
        raise ValueError(
            f"Not enough unique dates ({n}) for splits={n_splits}, test_size={test_size}, "
            f"min_train={min_train_size}, embargo={embargo}."
        )

    start = n - required
    for i in range(n_splits):
        test_start = start + i * test_size
        test_end = min(test_start + test_size, n)
        train_end = test_start - embargo
        if train_end < min_train_size:
            continue
        train_dates = unique_dates[:train_end]
        test_dates = unique_dates[test_start:test_end]
        yield i + 1, train_dates, test_dates


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    pred_sign = np.sign(y_pred)
    true_sign = np.sign(y_true)
    directional_accuracy = float(np.mean(pred_sign == true_sign))

    strategy_ret = pred_sign * y_true
    std = float(np.std(strategy_ret, ddof=1)) if len(strategy_ret) > 1 else np.nan
    sharpe = float(np.sqrt(252) * np.mean(strategy_ret) / std) if np.isfinite(std) and std > 0 else float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": directional_accuracy,
        "sharpe": sharpe,
    }


def downsample_xy(X: pd.DataFrame, y: np.ndarray, max_rows: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    if max_rows <= 0 or len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_rows, replace=False)
    idx = np.sort(idx)
    return X.iloc[idx], y[idx]


def build_lagged_panel_dataset(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    lag_window: int,
) -> pd.DataFrame:
    if lag_window < 0:
        raise ValueError("lag_window must be >= 0")

    keep = ["Date", "Symbol", target] + features
    work = df[keep].dropna(subset=[target]).copy()
    work = work.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    if lag_window == 0:
        return work

    lagged_blocks = []
    grp = work.groupby("Symbol", sort=False)
    for lag in range(1, lag_window + 1):
        shifted = grp[features].shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in features]
        lagged_blocks.append(shifted)

    out = pd.concat([work] + lagged_blocks, axis=1)
    return out


def make_model_builder(model_family: str, epochs: int, patience: int) -> Pipeline:
    if model_family == "elastic_net":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler(quantile_range=(5.0, 95.0))),
                ("model", ElasticNet(alpha=0.0005, l1_ratio=0.25, max_iter=6000, random_state=42)),
            ]
        )

    if model_family == "ridge":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler(quantile_range=(5.0, 95.0))),
                ("model", Ridge(alpha=1.0, random_state=42)),
            ]
        )

    if model_family == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        max_depth=12,
                        min_samples_leaf=5,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if model_family == "hist_gbr":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_depth=6,
                        max_iter=max(100, epochs),
                        early_stopping=True,
                        n_iter_no_change=max(5, patience),
                        validation_fraction=0.1,
                        random_state=42,
                    ),
                ),
            ]
        )

    if model_family == "mlp":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=max(100, epochs),
                        early_stopping=True,
                        n_iter_no_change=max(5, patience),
                        validation_fraction=0.1,
                        random_state=42,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unsupported model family: {model_family}")


def evaluate_model_on_panel(
    work: pd.DataFrame,
    feature_cols: list[str],
    model_family: str,
    epochs: int,
    patience: int,
    n_splits: int,
    test_size: int,
    min_train_size: int,
    embargo: int,
    max_train_rows: int,
    max_test_rows: int,
    seed: int,
) -> dict:
    uniq_dates = pd.DatetimeIndex(sorted(work["Date"].dropna().unique()))

    fold_rows = []
    y_true_all = []
    y_pred_all = []

    for fold_id, train_dates, test_dates in walkforward_date_splits(
        unique_dates=uniq_dates,
        n_splits=n_splits,
        test_size=test_size,
        min_train_size=min_train_size,
        embargo=embargo,
    ):
        train_mask = work["Date"].isin(train_dates)
        test_mask = work["Date"].isin(test_dates)

        X_train = work.loc[train_mask, feature_cols]
        y_train = work.loc[train_mask, "target"].astype(float).values
        X_test = work.loc[test_mask, feature_cols]
        y_test = work.loc[test_mask, "target"].astype(float).values

        X_train, y_train = downsample_xy(X_train, y_train, max_rows=max_train_rows, seed=seed + fold_id)
        X_test, y_test = downsample_xy(X_test, y_test, max_rows=max_test_rows, seed=seed + 100 + fold_id)

        model = make_model_builder(model_family=model_family, epochs=epochs, patience=patience)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_metrics = compute_metrics(y_test, y_pred)
        fold_rows.append(
            {
                "fold": fold_id,
                **fold_metrics,
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
            }
        )

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

    if not fold_rows:
        raise RuntimeError("No valid panel folds were created. Adjust split parameters.")

    yt = np.concatenate(y_true_all)
    yp = np.concatenate(y_pred_all)
    overall = compute_metrics(yt, yp)

    return {
        "overall": overall,
        "n_rows": int(len(work)),
        "n_features": int(len(feature_cols)),
        "n_test_total": int(len(yt)),
    }


def fit_final_and_dump(
    work: pd.DataFrame,
    feature_cols: list[str],
    model_family: str,
    epochs: int,
    patience: int,
    artifact_path: Path,
    max_train_rows: int,
    seed: int,
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    X = work[feature_cols]
    y = work["target"].astype(float).values
    X, y = downsample_xy(X, y, max_rows=max_train_rows, seed=seed)

    model = make_model_builder(model_family=model_family, epochs=epochs, patience=patience)
    model.fit(X, y)
    dump(model, artifact_path)


def model_status_row(model_name: str, status: str, note: str) -> dict:
    return {
        "model": model_name,
        "status": status,
        "rmse": np.nan,
        "mae": np.nan,
        "directional_accuracy": np.nan,
        "sharpe": np.nan,
        "n_features": 0,
        "n_rows": 0,
        "n_test_total": 0,
        "note": note,
    }


def write_summary_md(group_df: pd.DataFrame, model_family_df: pd.DataFrame, model_df: pd.DataFrame, out: Path, target: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Panel Walk-Forward Experiment Summary",
        "",
        f"- Target: {target}",
        "",
        "## Feature Group Results",
    ]

    for _, row in group_df.sort_values("rmse").iterrows():
        lines.append(
            f"- {row['feature_group']}: rmse={row['rmse']:.6f}, mae={row['mae']:.6f}, "
            f"dir_acc={row['directional_accuracy']:.4f}, sharpe={row['sharpe']:.4f}, "
            f"n_features={int(row['n_features'])}"
        )

    lines.extend(["", "## Model Family Sweep (Top 20 by RMSE)"])
    for _, row in model_family_df[model_family_df["status"] == "ok"].sort_values("rmse").head(20).iterrows():
        lines.append(
            f"- {row['model_family']} | {row['feature_group']} | lag={int(row['lag_window'])} | "
            f"epochs={int(row['epochs'])} | patience={int(row['patience'])}: "
            f"rmse={row['rmse']:.6f}, mae={row['mae']:.6f}, dir_acc={row['directional_accuracy']:.4f}, "
            f"sharpe={row['sharpe']:.4f}"
        )

    lines.extend(["", "## Model Set A-E"])
    for _, row in model_df.iterrows():
        if row["status"] != "ok":
            lines.append(f"- {row['model']}: {row['status']} ({row['note']})")
        else:
            lines.append(
                f"- {row['model']}: rmse={row['rmse']:.6f}, mae={row['mae']:.6f}, "
                f"dir_acc={row['directional_accuracy']:.4f}, sharpe={row['sharpe']:.4f}, "
                f"family={row['model_family']}, lag={int(row['lag_window'])}, epochs={int(row['epochs'])}, "
                f"patience={int(row['patience'])}, n_features={int(row['n_features'])}"
            )

    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {args.features_csv}")

    df = pd.read_csv(args.features_csv)
    if "Symbol" not in df.columns:
        raise ValueError("Panel features must include Symbol column.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values(["Date", "Symbol"]).reset_index(drop=True)

    g1_cols = [c for c in df.columns if c.startswith("g1_")]
    g2_cols = [c for c in df.columns if c.startswith("g2_")]
    g3_cols = [c for c in df.columns if c.startswith("g3_")]
    sentiment_cols = [c for c in df.columns if c.startswith(args.sentiment_prefix)]

    model_families = parse_csv_list(args.models)
    supported = set(DEFAULT_MODEL_FAMILIES)
    unknown = sorted(set(model_families).difference(supported))
    if unknown:
        raise ValueError(f"Unsupported model families: {unknown}")

    if args.tune:
        lag_windows = parse_int_grid(args.lag_window_grid)
        epochs_grid = parse_int_grid(args.epochs_grid)
        patience_grid = parse_int_grid(args.patience_grid)
    else:
        lag_windows = [args.lag_window]
        epochs_grid = [args.epochs]
        patience_grid = [args.patience]

    feature_groups = {
        "G1_price_only": g1_cols,
        "G2_price_technical": g1_cols + g2_cols,
        "G3_plus_panel_breadth": g1_cols + g2_cols + g3_cols,
    }

    model_family_rows: list[dict] = []

    def run_group_sweep(group_name: str, run_df: pd.DataFrame, base_cols: list[str], seed_offset: int = 0):
        for lag_window in lag_windows:
            lagged = build_lagged_panel_dataset(run_df, features=base_cols, target=args.target, lag_window=lag_window)
            lagged = lagged.rename(columns={args.target: "target"})
            feature_cols = [c for c in lagged.columns if c not in {"Date", "Symbol", "target"}]

            for model_family in model_families:
                if model_family in {"mlp", "hist_gbr"}:
                    grid = list(product(epochs_grid, patience_grid))
                else:
                    grid = [(args.epochs, args.patience)]

                for epochs, patience in grid:
                    try:
                        result = evaluate_model_on_panel(
                            work=lagged,
                            feature_cols=feature_cols,
                            model_family=model_family,
                            epochs=epochs,
                            patience=patience,
                            n_splits=args.n_splits,
                            test_size=args.test_size,
                            min_train_size=args.min_train_size,
                            embargo=args.embargo,
                            max_train_rows=args.max_train_rows,
                            max_test_rows=args.max_test_rows,
                            seed=args.shuffle_seed + seed_offset,
                        )

                        artifact_name = (
                            f"{args.target}_{group_name}_{model_family}_"
                            f"lag{lag_window}_e{epochs}_p{patience}.joblib"
                        ).replace("/", "_")
                        artifact_path = args.model_artifacts_dir / args.target / artifact_name
                        fit_final_and_dump(
                            work=lagged,
                            feature_cols=feature_cols,
                            model_family=model_family,
                            epochs=epochs,
                            patience=patience,
                            artifact_path=artifact_path,
                            max_train_rows=args.max_train_rows,
                            seed=args.shuffle_seed + 777 + seed_offset,
                        )

                        model_family_rows.append(
                            {
                                "feature_group": group_name,
                                "model_family": model_family,
                                "lag_window": lag_window,
                                "epochs": epochs,
                                "patience": patience,
                                **result["overall"],
                                "n_features": result["n_features"],
                                "n_rows": result["n_rows"],
                                "n_test_total": result["n_test_total"],
                                "status": "ok",
                                "note": "",
                                "artifact_path": str(artifact_path),
                            }
                        )
                    except Exception as exc:
                        model_family_rows.append(
                            {
                                "feature_group": group_name,
                                "model_family": model_family,
                                "lag_window": lag_window,
                                "epochs": epochs,
                                "patience": patience,
                                "rmse": np.nan,
                                "mae": np.nan,
                                "directional_accuracy": np.nan,
                                "sharpe": np.nan,
                                "n_features": len(feature_cols),
                                "n_rows": len(lagged),
                                "n_test_total": 0,
                                "status": "error",
                                "note": str(exc),
                                "artifact_path": "",
                            }
                        )

    for group_name, cols in feature_groups.items():
        run_group_sweep(group_name=group_name, run_df=df, base_cols=cols)

    model_family_df = pd.DataFrame(model_family_rows)
    args.model_family_results.parent.mkdir(parents=True, exist_ok=True)
    model_family_df.to_csv(args.model_family_results, index=False)

    # Best configuration per feature group.
    best_group_rows = []
    for group_name in feature_groups:
        ok = model_family_df[(model_family_df["feature_group"] == group_name) & (model_family_df["status"] == "ok")].copy()
        if ok.empty:
            continue
        best = ok.sort_values("rmse").iloc[0]
        best_group_rows.append(
            {
                "feature_group": group_name,
                "best_model_family": best["model_family"],
                "lag_window": int(best["lag_window"]),
                "epochs": int(best["epochs"]),
                "patience": int(best["patience"]),
                "rmse": float(best["rmse"]),
                "mae": float(best["mae"]),
                "directional_accuracy": float(best["directional_accuracy"]),
                "sharpe": float(best["sharpe"]),
                "n_features": int(best["n_features"]),
                "n_rows": int(best["n_rows"]),
                "n_test_total": int(best["n_test_total"]),
                "artifact_path": best["artifact_path"],
            }
        )

    group_df = pd.DataFrame(best_group_rows)
    args.group_results.parent.mkdir(parents=True, exist_ok=True)
    group_df.to_csv(args.group_results, index=False)

    def best_for_group(group_name: str) -> pd.Series | None:
        part = model_family_df[(model_family_df["feature_group"] == group_name) & (model_family_df["status"] == "ok")].copy()
        if part.empty:
            return None
        return part.sort_values("rmse").iloc[0]

    model_rows: list[dict] = []

    best_g1 = best_for_group("G1_price_only")
    if best_g1 is not None:
        model_rows.append(
            {
                "model": "Model_A_price_only",
                "status": "ok",
                "model_family": best_g1["model_family"],
                "lag_window": int(best_g1["lag_window"]),
                "epochs": int(best_g1["epochs"]),
                "patience": int(best_g1["patience"]),
                "rmse": float(best_g1["rmse"]),
                "mae": float(best_g1["mae"]),
                "directional_accuracy": float(best_g1["directional_accuracy"]),
                "sharpe": float(best_g1["sharpe"]),
                "n_features": int(best_g1["n_features"]),
                "n_rows": int(best_g1["n_rows"]),
                "n_test_total": int(best_g1["n_test_total"]),
                "note": "best configuration on G1",
                "artifact_path": best_g1["artifact_path"],
            }
        )
    else:
        model_rows.append({**model_status_row("Model_A_price_only", "not_run", "No successful G1 runs."), "model_family": "", "lag_window": 0, "epochs": 0, "patience": 0, "artifact_path": ""})

    best_g3 = best_for_group("G3_plus_panel_breadth")
    if best_g3 is not None:
        model_rows.append(
            {
                "model": "Model_B_price_full_available",
                "status": "ok",
                "model_family": best_g3["model_family"],
                "lag_window": int(best_g3["lag_window"]),
                "epochs": int(best_g3["epochs"]),
                "patience": int(best_g3["patience"]),
                "rmse": float(best_g3["rmse"]),
                "mae": float(best_g3["mae"]),
                "directional_accuracy": float(best_g3["directional_accuracy"]),
                "sharpe": float(best_g3["sharpe"]),
                "n_features": int(best_g3["n_features"]),
                "n_rows": int(best_g3["n_rows"]),
                "n_test_total": int(best_g3["n_test_total"]),
                "note": "best configuration on G1+G2+G3 panel features",
                "artifact_path": best_g3["artifact_path"],
            }
        )
    else:
        model_rows.append({**model_status_row("Model_B_price_full_available", "not_run", "No successful G3 runs."), "model_family": "", "lag_window": 0, "epochs": 0, "patience": 0, "artifact_path": ""})

    if sentiment_cols:
        run_group_sweep(group_name="G5_sentiment_only", run_df=df, base_cols=sentiment_cols, seed_offset=10)

        shuffled_df = df.copy()
        rng = np.random.default_rng(args.shuffle_seed)
        for col in sentiment_cols:
            shuffled_df[col] = rng.permutation(shuffled_df[col].to_numpy())
        run_group_sweep(
            group_name="G3_plus_shuffled_sentiment",
            run_df=shuffled_df,
            base_cols=g1_cols + g2_cols + g3_cols + sentiment_cols,
            seed_offset=20,
        )

        lagged_sent_df = df.copy()
        lagged_sent_df = lagged_sent_df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
        lagged_sent_df[sentiment_cols] = lagged_sent_df.groupby("Symbol", sort=False)[sentiment_cols].shift(1)
        run_group_sweep(
            group_name="G3_plus_lagged_sentiment",
            run_df=lagged_sent_df,
            base_cols=g1_cols + g2_cols + g3_cols + sentiment_cols,
            seed_offset=30,
        )

        # Refresh frame after appending custom groups.
        model_family_df = pd.DataFrame(model_family_rows)
        args.model_family_results.parent.mkdir(parents=True, exist_ok=True)
        model_family_df.to_csv(args.model_family_results, index=False)

        for model_name, group_name, note in [
            (
                "Model_C_sentiment_only",
                "G5_sentiment_only",
                "Phase 8 baseline if using FinBERT prefix; challenger if using RAG prefix.",
            ),
            (
                "Model_D_price_shuffled_sentiment",
                "G3_plus_shuffled_sentiment",
                "price + panel breadth + shuffled sentiment control",
            ),
            (
                "Model_E_price_lagged_sentiment",
                "G3_plus_lagged_sentiment",
                "price + panel breadth + lagged sentiment control (t-1 by symbol)",
            ),
        ]:
            best = best_for_group(group_name)
            if best is None:
                model_rows.append({**model_status_row(model_name, "not_run", f"No successful runs for {group_name}."), "model_family": "", "lag_window": 0, "epochs": 0, "patience": 0, "artifact_path": ""})
                continue

            model_rows.append(
                {
                    "model": model_name,
                    "status": "ok",
                    "model_family": best["model_family"],
                    "lag_window": int(best["lag_window"]),
                    "epochs": int(best["epochs"]),
                    "patience": int(best["patience"]),
                    "rmse": float(best["rmse"]),
                    "mae": float(best["mae"]),
                    "directional_accuracy": float(best["directional_accuracy"]),
                    "sharpe": float(best["sharpe"]),
                    "n_features": int(best["n_features"]),
                    "n_rows": int(best["n_rows"]),
                    "n_test_total": int(best["n_test_total"]),
                    "note": note,
                    "artifact_path": best["artifact_path"],
                }
            )
    else:
        for model_name in ["Model_C_sentiment_only", "Model_D_price_shuffled_sentiment", "Model_E_price_lagged_sentiment"]:
            model_rows.append(
                {
                    **model_status_row(
                        model_name=model_name,
                        status="not_run",
                        note=f"No sentiment features with prefix {args.sentiment_prefix} were found.",
                    ),
                    "model_family": "",
                    "lag_window": 0,
                    "epochs": 0,
                    "patience": 0,
                    "artifact_path": "",
                }
            )

    model_df = pd.DataFrame(model_rows)
    args.model_results.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(args.model_results, index=False)

    summary = {
        "target": args.target,
        "rows": int(len(df)),
        "symbols": int(df["Symbol"].nunique()),
        "unique_dates": int(df["Date"].nunique()),
        "tune": bool(args.tune),
        "lag_windows": lag_windows,
        "epochs_grid": epochs_grid,
        "patience_grid": patience_grid,
        "model_families": model_families,
        "features": {
            "g1": len(g1_cols),
            "g2": len(g2_cols),
            "g3": len(g3_cols),
            "sentiment": len(sentiment_cols),
        },
        "feature_group_results": group_df.to_dict(orient="records"),
        "model_family_results": model_family_df.to_dict(orient="records"),
        "model_set_results": model_df.to_dict(orient="records"),
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_summary_md(group_df=group_df, model_family_df=model_family_df, model_df=model_df, out=args.summary_md, target=args.target)

    print("Panel walk-forward experiments complete.")
    print(f"Target: {args.target}")
    print(f"Rows: {len(df)} | Symbols: {df['Symbol'].nunique()} | Unique dates: {df['Date'].nunique()}")
    print(f"Model-family results: {args.model_family_results}")
    print(f"Model-set results: {args.model_results}")
    print(f"Summary: {args.summary_md}")


if __name__ == "__main__":
    main()
