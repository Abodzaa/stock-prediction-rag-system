#!/usr/bin/env python3
"""Run walk-forward experiments with model family sweeps and tunable windows."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Callable

from joblib import dump
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES_CSV = Path("data/processed/features/research_features_daily.csv")
DEFAULT_GROUP_RESULTS = Path("reports/feature_group_results.csv")
DEFAULT_MODEL_RESULTS = Path("reports/model_set_results.csv")
DEFAULT_MODEL_FAMILY_RESULTS = Path("reports/model_family_results.csv")
DEFAULT_SUMMARY_MD = Path("reports/experiment_walkforward_summary.md")
DEFAULT_SUMMARY_JSON = Path("reports/experiment_walkforward_summary.json")
DEFAULT_MODEL_ARTIFACTS_DIR = Path("artifacts/models")

DEFAULT_MODEL_FAMILIES = [
    "elastic_net",
    "ridge",
    "random_forest",
    "hist_gbr",
    "mlp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward baseline experiment suite.")
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--group-results", type=Path, default=DEFAULT_GROUP_RESULTS)
    parser.add_argument("--model-results", type=Path, default=DEFAULT_MODEL_RESULTS)
    parser.add_argument("--model-family-results", type=Path, default=DEFAULT_MODEL_FAMILY_RESULTS)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--model-artifacts-dir", type=Path, default=DEFAULT_MODEL_ARTIFACTS_DIR)
    parser.add_argument("--target", default="target_h1", choices=["target_h1", "target_h5"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=int, default=252)
    parser.add_argument("--min-train-size", type=int, default=1260)
    parser.add_argument("--embargo", type=int, default=5)
    parser.add_argument("--models", default=",".join(DEFAULT_MODEL_FAMILIES))
    parser.add_argument("--lag-window", type=int, default=20)
    parser.add_argument("--lag-window-grid", default="10,20,60")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--epochs-grid", default="150,300")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--patience-grid", default="10,20")
    parser.add_argument("--sentiment-prefix", default="g5_sent_")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--tune", action="store_true")
    return parser.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value.")
    return items


def parse_int_grid(raw: str) -> list[int]:
    vals = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not vals:
        raise ValueError("Expected at least one integer in grid.")
    return vals


def walkforward_splits(n: int, n_splits: int, test_size: int, min_train_size: int, embargo: int):
    required = n_splits * test_size
    if n <= required + min_train_size + embargo:
        raise ValueError(
            f"Not enough rows ({n}) for requested walk-forward setup "
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


def build_lagged_dataset(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    lag_window: int,
) -> pd.DataFrame:
    if lag_window < 0:
        raise ValueError("lag_window must be >= 0")

    use_cols = ["Date", target] + features
    work = df[use_cols].dropna(subset=[target]).copy()
    work = work.sort_values("Date").reset_index(drop=True)

    if lag_window == 0:
        return work

    lag_blocks = []
    for lag in range(1, lag_window + 1):
        shifted = work[features].shift(lag).copy()
        shifted.columns = [f"{c}_lag{lag}" for c in features]
        lag_blocks.append(shifted)

    return pd.concat([work] + lag_blocks, axis=1)


def make_model_builder(
    model_family: str,
    epochs: int,
    patience: int,
) -> Callable[[], Pipeline]:
    if model_family == "elastic_net":
        return lambda: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.0005, l1_ratio=0.2, max_iter=5000, random_state=42)),
            ]
        )

    if model_family == "ridge":
        return lambda: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=42)),
            ]
        )

    if model_family == "random_forest":
        return lambda: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        max_depth=8,
                        min_samples_leaf=5,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if model_family == "hist_gbr":
        return lambda: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_depth=4,
                        max_iter=max(50, epochs),
                        early_stopping=True,
                        n_iter_no_change=max(5, patience),
                        validation_fraction=0.1,
                        random_state=42,
                    ),
                ),
            ]
        )

    if model_family == "mlp":
        return lambda: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=max(50, epochs),
                        early_stopping=True,
                        n_iter_no_change=max(5, patience),
                        validation_fraction=0.1,
                        random_state=42,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unsupported model family: {model_family}")


def evaluate_model_on_dataset(
    work: pd.DataFrame,
    feature_cols: list[str],
    model_builder: Callable[[], Pipeline],
    n_splits: int,
    test_size: int,
    min_train_size: int,
    embargo: int,
) -> dict:
    X_all = work[feature_cols]
    y_all = work["target"].astype(float).values

    fold_rows = []
    y_true_parts = []
    y_pred_parts = []

    for fold_id, train_start, train_end, test_start, test_end in walkforward_splits(
        n=len(work),
        n_splits=n_splits,
        test_size=test_size,
        min_train_size=min_train_size,
        embargo=embargo,
    ):
        X_train = X_all.iloc[train_start:train_end]
        y_train = y_all[train_start:train_end]
        X_test = X_all.iloc[test_start:test_end]
        y_test = y_all[test_start:test_end]

        model = model_builder()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_metrics = compute_metrics(y_true=y_test, y_pred=y_pred)
        fold_rows.append({"fold": fold_id, **fold_metrics, "n_test": int(len(y_test))})

        y_true_parts.append(y_test)
        y_pred_parts.append(y_pred)

    if not fold_rows:
        raise RuntimeError("No valid folds were created. Adjust split parameters.")

    y_true_all = np.concatenate(y_true_parts)
    y_pred_all = np.concatenate(y_pred_parts)
    overall = compute_metrics(y_true_all, y_pred_all)

    fold_df = pd.DataFrame(fold_rows)
    return {
        "n_rows": int(len(work)),
        "n_features": int(len(feature_cols)),
        "fold_metrics": fold_df,
        "overall": overall,
        "n_test_total": int(len(y_true_all)),
    }


def fit_final_model_and_dump(
    work: pd.DataFrame,
    feature_cols: list[str],
    model_builder: Callable[[], Pipeline],
    artifact_path: Path,
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    final_model = model_builder()
    final_model.fit(work[feature_cols], work["target"].astype(float).values)
    dump(final_model, artifact_path)


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


def write_markdown_summary(
    group_df: pd.DataFrame,
    model_family_df: pd.DataFrame,
    model_df: pd.DataFrame,
    summary_md: Path,
    target: str,
) -> None:
    summary_md.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Walk-Forward Experiment Summary",
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

    lines.extend(["", "## Model Family Sweep (Top 12 by RMSE)"])
    for _, row in model_family_df.sort_values("rmse").head(12).iterrows():
        lines.append(
            f"- {row['model_family']} | {row['feature_group']} | lag={int(row['lag_window'])} | "
            f"epochs={int(row['epochs'])} | patience={int(row['patience'])}: "
            f"rmse={row['rmse']:.6f}, mae={row['mae']:.6f}, "
            f"dir_acc={row['directional_accuracy']:.4f}, sharpe={row['sharpe']:.4f}"
        )

    lines.extend(["", "## Model Set A-E Results"]) 
    for _, row in model_df.iterrows():
        if row["status"] != "ok":
            lines.append(f"- {row['model']}: {row['status']} ({row['note']})")
        else:
            lines.append(
                f"- {row['model']}: rmse={row['rmse']:.6f}, mae={row['mae']:.6f}, "
                f"dir_acc={row['directional_accuracy']:.4f}, sharpe={row['sharpe']:.4f}, "
                f"n_features={int(row['n_features'])}, model_family={row['model_family']}, "
                f"lag={int(row['lag_window'])}, epochs={int(row['epochs'])}, patience={int(row['patience'])}"
            )

    lines.append("")
    summary_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {args.features_csv}")

    df = pd.read_csv(args.features_csv)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

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
        "G3_plus_breadth": g1_cols + g2_cols + g3_cols,
    }

    lagged_cache: dict[tuple[str, int], tuple[pd.DataFrame, list[str]]] = {}

    model_family_rows = []
    for feature_group, base_cols in feature_groups.items():
        for lag_window in lag_windows:
            lagged = build_lagged_dataset(df=df, features=base_cols, target=args.target, lag_window=lag_window)
            lagged = lagged.rename(columns={args.target: "target"})
            feature_cols = [c for c in lagged.columns if c not in {"Date", "target"}]
            lagged_cache[(feature_group, lag_window)] = (lagged, feature_cols)

            for model_family in model_families:
                if model_family in {"mlp", "hist_gbr"}:
                    param_grid = list(product(epochs_grid, patience_grid))
                else:
                    param_grid = [(args.epochs, args.patience)]

                for epochs, patience in param_grid:
                    builder = make_model_builder(model_family=model_family, epochs=epochs, patience=patience)
                    try:
                        result = evaluate_model_on_dataset(
                            work=lagged,
                            feature_cols=feature_cols,
                            model_builder=builder,
                            n_splits=args.n_splits,
                            test_size=args.test_size,
                            min_train_size=args.min_train_size,
                            embargo=args.embargo,
                        )

                        artifact_name = (
                            f"{args.target}_{feature_group}_{model_family}_"
                            f"lag{lag_window}_e{epochs}_p{patience}.joblib"
                        ).replace("/", "_")
                        artifact_path = args.model_artifacts_dir / args.target / artifact_name
                        fit_final_model_and_dump(
                            work=lagged,
                            feature_cols=feature_cols,
                            model_builder=builder,
                            artifact_path=artifact_path,
                        )

                        model_family_rows.append(
                            {
                                "feature_group": feature_group,
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
                                "feature_group": feature_group,
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

    model_family_df = pd.DataFrame(model_family_rows)
    args.model_family_results.parent.mkdir(parents=True, exist_ok=True)
    model_family_df.to_csv(args.model_family_results, index=False)

    best_group_rows = []
    for feature_group in feature_groups:
        subset = model_family_df[
            (model_family_df["feature_group"] == feature_group)
            & (model_family_df["status"] == "ok")
        ].copy()
        if subset.empty:
            continue
        best = subset.sort_values("rmse").iloc[0]
        best_group_rows.append(
            {
                "feature_group": feature_group,
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

    def run_custom_sweep(
        group_name: str,
        run_df: pd.DataFrame,
        base_cols: list[str],
    ) -> tuple[pd.Series | None, list[dict]]:
        custom_rows: list[dict] = []

        for lag_window in lag_windows:
            lagged = build_lagged_dataset(
                df=run_df,
                features=base_cols,
                target=args.target,
                lag_window=lag_window,
            )
            lagged = lagged.rename(columns={args.target: "target"})
            feature_cols = [c for c in lagged.columns if c not in {"Date", "target"}]

            for model_family in model_families:
                if model_family in {"mlp", "hist_gbr"}:
                    param_grid = list(product(epochs_grid, patience_grid))
                else:
                    param_grid = [(args.epochs, args.patience)]

                for epochs, patience in param_grid:
                    builder = make_model_builder(model_family=model_family, epochs=epochs, patience=patience)
                    try:
                        result = evaluate_model_on_dataset(
                            work=lagged,
                            feature_cols=feature_cols,
                            model_builder=builder,
                            n_splits=args.n_splits,
                            test_size=args.test_size,
                            min_train_size=args.min_train_size,
                            embargo=args.embargo,
                        )

                        artifact_name = (
                            f"{args.target}_{group_name}_{model_family}_"
                            f"lag{lag_window}_e{epochs}_p{patience}.joblib"
                        ).replace("/", "_")
                        artifact_path = args.model_artifacts_dir / args.target / artifact_name
                        fit_final_model_and_dump(
                            work=lagged,
                            feature_cols=feature_cols,
                            model_builder=builder,
                            artifact_path=artifact_path,
                        )

                        custom_rows.append(
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
                        custom_rows.append(
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

        if not custom_rows:
            return None, custom_rows

        ok_rows = [r for r in custom_rows if r["status"] == "ok"]
        if not ok_rows:
            return None, custom_rows

        best = pd.DataFrame(ok_rows).sort_values("rmse").iloc[0]
        return best, custom_rows

    # Required model set A-E.
    model_rows = []

    def best_for_group(group_name: str) -> pd.Series | None:
        s = model_family_df[
            (model_family_df["feature_group"] == group_name)
            & (model_family_df["status"] == "ok")
        ].copy()
        if s.empty:
            return None
        return s.sort_values("rmse").iloc[0]

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
        model_rows.append(
            {
                **model_status_row(
                    model_name="Model_A_price_only",
                    status="not_run",
                    note="No successful runs for G1.",
                ),
                "model_family": "",
                "lag_window": 0,
                "epochs": 0,
                "patience": 0,
                "artifact_path": "",
            }
        )

    best_g3 = best_for_group("G3_plus_breadth")
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
                "note": "best configuration on G1+G2+G3 available in workspace",
                "artifact_path": best_g3["artifact_path"],
            }
        )
    else:
        model_rows.append(
            {
                **model_status_row(
                    model_name="Model_B_price_full_available",
                    status="not_run",
                    note="No successful runs for G3.",
                ),
                "model_family": "",
                "lag_window": 0,
                "epochs": 0,
                "patience": 0,
                "artifact_path": "",
            }
        )

    if sentiment_cols:
        best_sent_only, sent_only_rows = run_custom_sweep(
            group_name="G5_sentiment_only",
            run_df=df,
            base_cols=sentiment_cols,
        )
        model_family_rows.extend(sent_only_rows)

        if best_sent_only is not None:
            model_rows.append(
                {
                    "model": "Model_C_sentiment_only",
                    "status": "ok",
                    "model_family": best_sent_only["model_family"],
                    "lag_window": int(best_sent_only["lag_window"]),
                    "epochs": int(best_sent_only["epochs"]),
                    "patience": int(best_sent_only["patience"]),
                    "rmse": float(best_sent_only["rmse"]),
                    "mae": float(best_sent_only["mae"]),
                    "directional_accuracy": float(best_sent_only["directional_accuracy"]),
                    "sharpe": float(best_sent_only["sharpe"]),
                    "n_features": int(best_sent_only["n_features"]),
                    "n_rows": int(best_sent_only["n_rows"]),
                    "n_test_total": int(best_sent_only["n_test_total"]),
                    "note": f"best configuration on sentiment-only features ({args.sentiment_prefix}*)",
                    "artifact_path": best_sent_only["artifact_path"],
                }
            )
        else:
            model_rows.append(
                {
                    **model_status_row(
                        model_name="Model_C_sentiment_only",
                        status="not_run",
                        note="Sentiment columns found but no successful sentiment-only runs.",
                    ),
                    "model_family": "",
                    "lag_window": 0,
                    "epochs": 0,
                    "patience": 0,
                    "artifact_path": "",
                }
            )

        shuffled_df = df.copy()
        rng = np.random.default_rng(args.shuffle_seed)
        for col in sentiment_cols:
            shuffled_df[col] = rng.permutation(shuffled_df[col].to_numpy())

        best_shuffled, shuffled_rows = run_custom_sweep(
            group_name="G3_plus_shuffled_sentiment",
            run_df=shuffled_df,
            base_cols=g1_cols + g2_cols + g3_cols + sentiment_cols,
        )
        model_family_rows.extend(shuffled_rows)

        if best_shuffled is not None:
            model_rows.append(
                {
                    "model": "Model_D_price_shuffled_sentiment",
                    "status": "ok",
                    "model_family": best_shuffled["model_family"],
                    "lag_window": int(best_shuffled["lag_window"]),
                    "epochs": int(best_shuffled["epochs"]),
                    "patience": int(best_shuffled["patience"]),
                    "rmse": float(best_shuffled["rmse"]),
                    "mae": float(best_shuffled["mae"]),
                    "directional_accuracy": float(best_shuffled["directional_accuracy"]),
                    "sharpe": float(best_shuffled["sharpe"]),
                    "n_features": int(best_shuffled["n_features"]),
                    "n_rows": int(best_shuffled["n_rows"]),
                    "n_test_total": int(best_shuffled["n_test_total"]),
                    "note": "price + breadth + shuffled sentiment control",
                    "artifact_path": best_shuffled["artifact_path"],
                }
            )
        else:
            model_rows.append(
                {
                    **model_status_row(
                        model_name="Model_D_price_shuffled_sentiment",
                        status="not_run",
                        note="Sentiment columns found but no successful shuffled-control runs.",
                    ),
                    "model_family": "",
                    "lag_window": 0,
                    "epochs": 0,
                    "patience": 0,
                    "artifact_path": "",
                }
            )

        lagged_sent_df = df.copy()
        lagged_sent_df[sentiment_cols] = lagged_sent_df[sentiment_cols].shift(1)

        best_lagged, lagged_rows = run_custom_sweep(
            group_name="G3_plus_lagged_sentiment",
            run_df=lagged_sent_df,
            base_cols=g1_cols + g2_cols + g3_cols + sentiment_cols,
        )
        model_family_rows.extend(lagged_rows)

        if best_lagged is not None:
            model_rows.append(
                {
                    "model": "Model_E_price_lagged_sentiment",
                    "status": "ok",
                    "model_family": best_lagged["model_family"],
                    "lag_window": int(best_lagged["lag_window"]),
                    "epochs": int(best_lagged["epochs"]),
                    "patience": int(best_lagged["patience"]),
                    "rmse": float(best_lagged["rmse"]),
                    "mae": float(best_lagged["mae"]),
                    "directional_accuracy": float(best_lagged["directional_accuracy"]),
                    "sharpe": float(best_lagged["sharpe"]),
                    "n_features": int(best_lagged["n_features"]),
                    "n_rows": int(best_lagged["n_rows"]),
                    "n_test_total": int(best_lagged["n_test_total"]),
                    "note": "price + breadth + lagged sentiment control (t-1)",
                    "artifact_path": best_lagged["artifact_path"],
                }
            )
        else:
            model_rows.append(
                {
                    **model_status_row(
                        model_name="Model_E_price_lagged_sentiment",
                        status="not_run",
                        note="Sentiment columns found but no successful lagged-control runs.",
                    ),
                    "model_family": "",
                    "lag_window": 0,
                    "epochs": 0,
                    "patience": 0,
                    "artifact_path": "",
                }
            )
    else:
        model_rows.append(
            {
                **model_status_row(
                    model_name="Model_C_sentiment_only",
                    status="not_run",
                    note=(
                        "No sentiment features with prefix "
                        f"{args.sentiment_prefix} were found in features CSV."
                    ),
                ),
                "model_family": "",
                "lag_window": 0,
                "epochs": 0,
                "patience": 0,
                "artifact_path": "",
            }
        )
        model_rows.append(
            {
                **model_status_row(
                    model_name="Model_D_price_shuffled_sentiment",
                    status="not_run",
                    note=(
                        "No sentiment features with prefix "
                        f"{args.sentiment_prefix} were found in features CSV."
                    ),
                ),
                "model_family": "",
                "lag_window": 0,
                "epochs": 0,
                "patience": 0,
                "artifact_path": "",
            }
        )
        model_rows.append(
            {
                **model_status_row(
                    model_name="Model_E_price_lagged_sentiment",
                    status="not_run",
                    note=(
                        "No sentiment features with prefix "
                        f"{args.sentiment_prefix} were found in features CSV."
                    ),
                ),
                "model_family": "",
                "lag_window": 0,
                "epochs": 0,
                "patience": 0,
                "artifact_path": "",
            }
        )

    model_family_df = pd.DataFrame(model_family_rows)
    args.model_family_results.parent.mkdir(parents=True, exist_ok=True)
    model_family_df.to_csv(args.model_family_results, index=False)

    model_df = pd.DataFrame(model_rows)
    args.model_results.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(args.model_results, index=False)

    summary_payload = {
        "target": args.target,
        "rows": int(len(df)),
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
    args.summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    write_markdown_summary(
        group_df=group_df,
        model_family_df=model_family_df,
        model_df=model_df,
        summary_md=args.summary_md,
        target=args.target,
    )

    print("Walk-forward experiments complete.")
    print(f"Target: {args.target}")
    print(f"Model-family sweep results written to: {args.model_family_results}")
    print(f"Feature groups written to: {args.group_results}")
    print(f"Model-set results written to: {args.model_results}")
    print(f"Model artifacts root: {args.model_artifacts_dir / args.target}")
    print(f"Summary markdown: {args.summary_md}")


if __name__ == "__main__":
    main()
