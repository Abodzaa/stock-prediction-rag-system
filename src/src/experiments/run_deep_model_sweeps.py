#!/usr/bin/env python3
"""Run deep-model sweeps for the plan's required architectures.

This runner is intentionally robust to partial dependency availability. Models whose
backend is unavailable are marked as `not_available` rather than crashing the full sweep.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deep_model_registry import ALL_MODELS, BASELINE_MODELS, PRIMARY_MODELS, build_hyperparameter_grid


DEFAULT_FEATURES_CSV = Path("data/processed/features/research_features_daily.csv")
DEFAULT_REPORT_CSV = Path("reports/deep_model_sweep_results.csv")
DEFAULT_SUMMARY_JSON = Path("reports/deep_model_sweep_summary.json")
DEFAULT_SUMMARY_MD = Path("reports/deep_model_sweep_summary.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep model sweep runner for required architecture list.")
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--target", default="target_h1", choices=["target_h1", "target_h5"])
    parser.add_argument("--models", default=",".join(ALL_MODELS))
    parser.add_argument("--input-sizes", default="64,128,256")
    parser.add_argument("--max-steps-grid", default="200,350")
    parser.add_argument("--patience-grid", default="10,20")
    parser.add_argument("--n-windows", type=int, default=5)
    parser.add_argument("--step-size", type=int, default=63)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    return parser.parse_args()


def parse_csv_items(raw: str) -> list[str]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value")
    return items


def parse_int_items(raw: str) -> list[int]:
    vals = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not vals:
        raise ValueError("Expected at least one integer value")
    return vals


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


def safe_constructor_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def get_neuralforecast_class(model_name: str):
    try:
        import neuralforecast.models as nfm
    except Exception:
        return None, "neuralforecast_not_installed"

    mapping = {
        "TFT": "TFT",
        "PatchTST": "PatchTST",
        "Mamba": "Mamba",
        "Informer": "Informer",
        "Autoformer": "Autoformer",
        "FEDformer": "FEDformer",
        "LSTM": "LSTM",
        "GRU": "GRU",
        "NBEATS": "NBEATS",
        "NHITS": "NHITS",
    }
    class_name = mapping[model_name]
    cls = getattr(nfm, class_name, None)
    if cls is None:
        return None, f"class_not_available:{class_name}"
    return cls, "ok"


def model_candidate_kwargs(model_name: str, h: int, spec) -> dict[str, Any]:
    common = {
        "h": h,
        "input_size": spec.input_size,
        "max_steps": spec.max_steps,
        "early_stop_patience_steps": spec.patience,
        "batch_size": spec.batch_size,
        "val_check_steps": 10,
        "learning_rate": 1e-3,
        "random_seed": 42,
    }

    model_specific = {
        "hidden_size": spec.hidden_size,
    }

    if model_name in {"PatchTST", "Informer", "Autoformer", "FEDformer"}:
        model_specific["n_heads"] = 4

    if model_name in {"NBEATS", "NHITS"}:
        model_specific["mlp_units"] = [[spec.hidden_size, spec.hidden_size]]

    common.update(model_specific)
    return common


def to_nf_dataset(df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> pd.DataFrame:
    work = df[["Date", target_col] + feature_cols].dropna(subset=[target_col]).copy()
    work = work.rename(columns={"Date": "ds", target_col: "y"})
    work["ds"] = pd.to_datetime(work["ds"], errors="coerce")
    work = work.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    work["unique_id"] = "SPX"

    cols = ["unique_id", "ds", "y"] + feature_cols
    return work[cols]


def evaluate_with_neuralforecast(
    model,
    dataset: pd.DataFrame,
    n_windows: int,
    step_size: int,
) -> dict[str, Any]:
    from neuralforecast import NeuralForecast

    nf = NeuralForecast(models=[model], freq="B")
    cv = nf.cross_validation(df=dataset, n_windows=n_windows, step_size=step_size, val_size=0, verbose=False)

    pred_cols = [c for c in cv.columns if c not in {"unique_id", "ds", "cutoff", "y"}]
    if not pred_cols:
        raise RuntimeError("No prediction column returned from NeuralForecast cross_validation")

    pred_col = pred_cols[0]
    y_true = cv["y"].to_numpy(dtype=float)
    y_pred = cv[pred_col].to_numpy(dtype=float)
    metrics = compute_metrics(y_true, y_pred)

    return {
        **metrics,
        "n_predictions": int(len(cv)),
        "pred_col": pred_col,
    }


def write_summary_md(
    summary_md: Path,
    primary_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    unavailable_df: pd.DataFrame,
) -> None:
    summary_md.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Deep Model Sweep Summary",
        "",
        "## Primary Models",
    ]

    if len(primary_df) == 0:
        lines.append("- none")
    else:
        for _, row in primary_df.sort_values("rmse").head(12).iterrows():
            lines.append(
                f"- {row['model_name']} | {row['feature_group']} | input={int(row['input_size'])} "
                f"| steps={int(row['max_steps'])} | patience={int(row['patience'])}: "
                f"rmse={row['rmse']:.6f}, dir_acc={row['directional_accuracy']:.4f}, sharpe={row['sharpe']:.4f}"
            )

    lines.extend(["", "## Baseline Models"])
    if len(baseline_df) == 0:
        lines.append("- none")
    else:
        for _, row in baseline_df.sort_values("rmse").head(12).iterrows():
            lines.append(
                f"- {row['model_name']} | {row['feature_group']} | input={int(row['input_size'])} "
                f"| steps={int(row['max_steps'])} | patience={int(row['patience'])}: "
                f"rmse={row['rmse']:.6f}, dir_acc={row['directional_accuracy']:.4f}, sharpe={row['sharpe']:.4f}"
            )

    lines.extend(["", "## Not Available / Failed"])
    if len(unavailable_df) == 0:
        lines.append("- none")
    else:
        for _, row in unavailable_df.head(20).iterrows():
            lines.append(
                f"- {row['model_name']} | {row['feature_group']} | status={row['status']} | note={row['note']}"
            )

    lines.append("")
    summary_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {args.features_csv}")

    requested_models = parse_csv_items(args.models)
    unknown = sorted(set(requested_models).difference(ALL_MODELS))
    if unknown:
        raise ValueError(f"Unknown models requested: {unknown}")

    input_sizes = parse_int_items(args.input_sizes)
    max_steps_grid = parse_int_items(args.max_steps_grid)
    patience_grid = parse_int_items(args.patience_grid)

    df = pd.read_csv(args.features_csv)

    g1 = [c for c in df.columns if c.startswith("g1_")]
    g2 = [c for c in df.columns if c.startswith("g2_")]
    g3 = [c for c in df.columns if c.startswith("g3_")]

    feature_groups = {
        "G1_price_only": g1,
        "G2_price_technical": g1 + g2,
        "G3_plus_breadth": g1 + g2 + g3,
    }

    rows = []

    for model_name in requested_models:
        model_cls, status_note = get_neuralforecast_class(model_name)

        for feature_group, feature_cols in feature_groups.items():
            specs = build_hyperparameter_grid(
                model_name=model_name,
                feature_group=feature_group,
                input_sizes=input_sizes,
                max_steps_grid=max_steps_grid,
                patience_grid=patience_grid,
            )

            if model_cls is None:
                for spec in specs:
                    rows.append(
                        {
                            "model_name": model_name,
                            "model_bucket": "primary" if model_name in PRIMARY_MODELS else "baseline",
                            "feature_group": feature_group,
                            "input_size": spec.input_size,
                            "max_steps": spec.max_steps,
                            "patience": spec.patience,
                            "hidden_size": spec.hidden_size,
                            "batch_size": spec.batch_size,
                            "status": "not_available",
                            "note": status_note,
                            "rmse": np.nan,
                            "mae": np.nan,
                            "directional_accuracy": np.nan,
                            "sharpe": np.nan,
                            "n_predictions": 0,
                        }
                    )
                continue

            dataset = to_nf_dataset(df=df, target_col=args.target, feature_cols=feature_cols)
            hist_exog_list = feature_cols if feature_cols else None

            for spec in specs:
                try:
                    kwargs = model_candidate_kwargs(model_name=model_name, h=1, spec=spec)
                    if hist_exog_list:
                        kwargs["hist_exog_list"] = hist_exog_list

                    kwargs = safe_constructor_kwargs(model_cls, kwargs)
                    model = model_cls(**kwargs)

                    result = evaluate_with_neuralforecast(
                        model=model,
                        dataset=dataset,
                        n_windows=args.n_windows,
                        step_size=args.step_size,
                    )

                    rows.append(
                        {
                            "model_name": model_name,
                            "model_bucket": "primary" if model_name in PRIMARY_MODELS else "baseline",
                            "feature_group": feature_group,
                            "input_size": spec.input_size,
                            "max_steps": spec.max_steps,
                            "patience": spec.patience,
                            "hidden_size": spec.hidden_size,
                            "batch_size": spec.batch_size,
                            "status": "ok",
                            "note": "",
                            "rmse": result["rmse"],
                            "mae": result["mae"],
                            "directional_accuracy": result["directional_accuracy"],
                            "sharpe": result["sharpe"],
                            "n_predictions": result["n_predictions"],
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "model_name": model_name,
                            "model_bucket": "primary" if model_name in PRIMARY_MODELS else "baseline",
                            "feature_group": feature_group,
                            "input_size": spec.input_size,
                            "max_steps": spec.max_steps,
                            "patience": spec.patience,
                            "hidden_size": spec.hidden_size,
                            "batch_size": spec.batch_size,
                            "status": "error",
                            "note": str(exc),
                            "rmse": np.nan,
                            "mae": np.nan,
                            "directional_accuracy": np.nan,
                            "sharpe": np.nan,
                            "n_predictions": 0,
                        }
                    )

    results = pd.DataFrame(rows)
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.report_csv, index=False)

    ok_df = results[results["status"] == "ok"].copy()
    primary_ok = ok_df[ok_df["model_bucket"] == "primary"].copy()
    baseline_ok = ok_df[ok_df["model_bucket"] == "baseline"].copy()
    unavailable = results[results["status"] != "ok"].copy()

    summary_payload = {
        "target": args.target,
        "models_requested": requested_models,
        "input_sizes": input_sizes,
        "max_steps_grid": max_steps_grid,
        "patience_grid": patience_grid,
        "counts": {
            "total_runs": int(len(results)),
            "ok": int((results["status"] == "ok").sum()),
            "not_available": int((results["status"] == "not_available").sum()),
            "error": int((results["status"] == "error").sum()),
        },
        "best_primary": primary_ok.sort_values("rmse").head(10).to_dict(orient="records"),
        "best_baseline": baseline_ok.sort_values("rmse").head(10).to_dict(orient="records"),
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    write_summary_md(
        summary_md=args.summary_md,
        primary_df=primary_ok,
        baseline_df=baseline_ok,
        unavailable_df=unavailable,
    )

    print("Deep model sweep finished.")
    print(f"Rows: {len(results)} | ok={(results['status'] == 'ok').sum()} ")
    print(f"Report CSV: {args.report_csv}")
    print(f"Summary JSON: {args.summary_json}")
    print(f"Summary MD: {args.summary_md}")


if __name__ == "__main__":
    main()
