#!/usr/bin/env python3
"""Run selected best deep-model configs on each symbol in a panel feature CSV."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.run_deep_torch_experiments import TrialSpec, evaluate_trial, pick_device


DEFAULT_PANEL_CSV = Path("data/processed/features/research_features_panel_nasdaq100.csv")
DEFAULT_BEST_SUMMARY = Path("reports/thesis_best_model_summary.csv")
DEFAULT_RESULTS_CSV = Path("reports/nasdaq100_best_deep_per_symbol_results.csv")
DEFAULT_ARTIFACT_MANIFEST_CSV = Path("reports/nasdaq100_best_deep_artifact_manifest.csv")
DEFAULT_SUMMARY_JSON = Path("reports/nasdaq100_best_deep_per_symbol_summary.json")
DEFAULT_SUMMARY_MD = Path("reports/nasdaq100_best_deep_per_symbol_summary.md")
DEFAULT_ARTIFACT_DIR = Path("artifacts/models/nasdaq100_best_deep_per_symbol")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run best S&P-derived deep configs per panel symbol.")
    parser.add_argument("--panel-csv", type=Path, default=DEFAULT_PANEL_CSV)
    parser.add_argument("--best-summary-csv", type=Path, default=DEFAULT_BEST_SUMMARY)
    parser.add_argument("--models", default="MambaStock,TFT")
    parser.add_argument("--targets", default="target_h1,target_h5")
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbol allowlist.")
    parser.add_argument("--max-symbols", type=int, default=0, help="0 means all symbols.")
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--artifact-manifest-csv", type=Path, default=DEFAULT_ARTIFACT_MANIFEST_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)

    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--test-size", type=int, default=126)
    parser.add_argument("--min-train-size", type=int, default=1260)
    parser.add_argument("--embargo", type=int, default=5)
    parser.add_argument("--val-size", type=int, default=252)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-errors", action="store_true", help="Retry existing error rows instead of skipping them.")
    return parser.parse_args()


def parse_csv_items(raw: str) -> list[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one comma-separated item.")
    return vals


def safe_int(value, default: int) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value, default: float) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def feature_groups_for(df: pd.DataFrame) -> dict[str, list[str]]:
    g1 = [c for c in df.columns if c.startswith("g1_")]
    g2 = [c for c in df.columns if c.startswith("g2_")]
    g3 = [c for c in df.columns if c.startswith("g3_")]
    g5 = [c for c in df.columns if c.startswith("g5_sent_")]

    groups = {
        "G1_price_only": g1,
        "G2_price_technical": g1 + g2,
        "G3_plus_breadth": g1 + g2 + g3,
        "G3_plus_panel_breadth": g1 + g2 + g3,
    }
    if g5:
        groups["G4_price_plus_sentiment"] = g1 + g2 + g3 + g5
        groups["G5_sentiment_only"] = g5
    return groups


def load_best_specs(best_summary_csv: Path, models: list[str], targets: list[str]) -> dict[tuple[str, str], pd.Series]:
    if not best_summary_csv.exists():
        raise FileNotFoundError(f"Best summary CSV not found: {best_summary_csv}")

    df = pd.read_csv(best_summary_csv)
    df = df[
        (df["selection_view"] == "best_per_model")
        & (df["experiment_scope"] == "deep_daily_price")
        & (df["model_name"].isin(models))
        & (df["target"].isin(targets))
    ].copy()
    if df.empty:
        raise RuntimeError("No matching deep_daily_price best_per_model rows found in summary CSV.")

    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
    df["directional_accuracy"] = pd.to_numeric(df["directional_accuracy"], errors="coerce")
    df["sharpe"] = pd.to_numeric(df["sharpe"], errors="coerce")
    df = df.sort_values(["rmse", "directional_accuracy", "sharpe"], ascending=[True, False, False])

    specs: dict[tuple[str, str], pd.Series] = {}
    for key, part in df.groupby(["model_name", "target"], sort=False):
        specs[(str(key[0]), str(key[1]))] = part.iloc[0]

    missing = [(m, t) for m in models for t in targets if (m, t) not in specs]
    if missing:
        raise RuntimeError(f"Missing best specs for model/target pairs: {missing}")
    return specs


def existing_completed(results_csv: Path, retry_errors: bool) -> set[tuple[str, str, str]]:
    if not results_csv.exists():
        return set()
    df = pd.read_csv(results_csv)
    if df.empty or not {"symbol", "model_name", "target"}.issubset(df.columns):
        return set()
    if retry_errors and "status" in df.columns:
        done = df[df["status"] == "ok"].copy()
    else:
        done = df
    return {(str(r["symbol"]), str(r["model_name"]), str(r["target"])) for _, r in done.iterrows()}


def write_csv_with_retry(df: pd.DataFrame, path_value: Path, index: bool = False) -> None:
    for attempt in range(1, 9):
        try:
            df.to_csv(path_value, index=index)
            return
        except (OSError, PermissionError):
            if attempt == 8:
                raise
            time.sleep(0.25 * attempt)


def write_text_with_retry(path_value: Path, text: str) -> None:
    for attempt in range(1, 9):
        try:
            path_value.write_text(text, encoding="utf-8")
            return
        except (OSError, PermissionError):
            if attempt == 8:
                raise
            time.sleep(0.25 * attempt)


def write_outputs(rows: list[dict], args: argparse.Namespace) -> None:
    df = pd.DataFrame(rows)
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)

    write_csv_with_retry(df, args.results_csv, index=False)

    ok = df[df["status"] == "ok"].copy() if not df.empty else pd.DataFrame()
    if not ok.empty:
        manifest_cols = [
            "symbol",
            "target",
            "model_name",
            "feature_group",
            "rmse",
            "mae",
            "directional_accuracy",
            "sharpe",
            "artifact_path",
            "source_best_report",
            "source_best_artifact",
            "n_splits",
            "test_size",
            "min_train_size",
            "embargo",
            "val_size",
            "lag_window",
            "hidden_size",
            "lr",
            "epochs",
            "patience",
            "batch_size",
        ]
        manifest_cols = [c for c in manifest_cols if c in ok.columns]
        write_csv_with_retry(ok[manifest_cols], args.artifact_manifest_csv, index=False)
    else:
        write_csv_with_retry(pd.DataFrame(), args.artifact_manifest_csv, index=False)

    for col in ["rmse", "directional_accuracy", "sharpe"]:
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")

    leaderboard = {}
    lines = [
        "# NASDAQ-100 Best Deep Models Per Symbol",
        "",
        "Selection source: S&P-derived best configs from thesis_best_model_summary.csv.",
        "Ranking: lowest RMSE, tie-break by higher directional accuracy then higher Sharpe.",
        "",
    ]

    if not ok.empty:
        grouped = ok.sort_values(["rmse", "directional_accuracy", "sharpe"], ascending=[True, False, False])
        for target in sorted(grouped["target"].unique()):
            lines.append(f"## {target}")
            target_part = grouped[grouped["target"] == target]
            leaderboard[target] = target_part.head(25).to_dict(orient="records")
            for _, r in target_part.head(10).iterrows():
                lines.append(
                    "- "
                    f"{r['symbol']} | {r['model_name']} | {r['feature_group']}: "
                    f"rmse={r['rmse']:.6f}, dir_acc={r['directional_accuracy']:.4f}, "
                    f"sharpe={r['sharpe']:.4f}"
                )
            lines.append("")
    else:
        lines.append("No successful rows yet.")

    payload = {
        "results_csv": str(args.results_csv),
        "artifact_manifest_csv": str(args.artifact_manifest_csv),
        "artifact_dir": str(args.artifact_dir),
        "evaluation_policy": {
            "n_splits": args.n_splits,
            "test_size": args.test_size,
            "min_train_size": args.min_train_size,
            "embargo": args.embargo,
            "val_size": args.val_size,
        },
        "rows": int(len(df)),
        "ok_rows": int(len(ok)),
        "leaderboard": leaderboard,
    }
    write_text_with_retry(args.summary_json, json.dumps(payload, indent=2))
    write_text_with_retry(args.summary_md, "\n".join(lines))


def main() -> None:
    args = parse_args()

    if not args.panel_csv.exists():
        raise FileNotFoundError(f"Panel CSV not found: {args.panel_csv}")

    models = parse_csv_items(args.models)
    targets = parse_csv_items(args.targets)
    best_specs = load_best_specs(args.best_summary_csv, models=models, targets=targets)

    panel = pd.read_csv(args.panel_csv)
    if "Symbol" not in panel.columns:
        raise ValueError("Panel CSV must include Symbol.")
    panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce")
    panel = panel.dropna(subset=["Date", "Symbol"]).sort_values(["Symbol", "Date"]).reset_index(drop=True)
    groups = feature_groups_for(panel)

    symbols = sorted(panel["Symbol"].astype(str).unique())
    if args.symbols.strip():
        allow = set(parse_csv_items(args.symbols))
        symbols = [s for s in symbols if s in allow]
    if args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]

    device = pick_device(args.device)
    completed = set() if args.overwrite else existing_completed(args.results_csv, retry_errors=args.retry_errors)
    rows = []
    if args.results_csv.exists() and not args.overwrite:
        rows = pd.read_csv(args.results_csv).to_dict(orient="records")

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    total_jobs = len(symbols) * len(targets) * len(models)
    job_number = 0
    print(
        f"Starting per-symbol deep run: symbols={len(symbols)} targets={len(targets)} "
        f"models={len(models)} total_jobs={total_jobs}",
        flush=True,
    )

    for symbol in symbols:
        symbol_df = panel[panel["Symbol"].astype(str) == symbol].copy()
        symbol_df = symbol_df.sort_values("Date").reset_index(drop=True)

        for target in targets:
            for model_name in models:
                job_number += 1
                key = (symbol, model_name, target)
                if key in completed:
                    print(f"[{job_number}/{total_jobs}] skip {symbol} {target} {model_name}: already ok", flush=True)
                    continue

                spec_row = best_specs[(model_name, target)]
                feature_group = str(spec_row["feature_group"])
                feature_cols = groups.get(feature_group)
                if not feature_cols:
                    raise RuntimeError(f"Feature group {feature_group} has no columns in {args.panel_csv}.")

                spec = TrialSpec(
                    model_name=model_name,
                    feature_group=feature_group,
                    lag_window=safe_int(spec_row.get("lag_window"), 64),
                    hidden_size=safe_int(spec_row.get("hidden_size"), 32),
                    lr=safe_float(spec_row.get("lr"), 1e-3),
                    epochs=safe_int(spec_row.get("epochs"), 20),
                    patience=safe_int(spec_row.get("patience"), 10),
                    batch_size=safe_int(spec_row.get("batch_size"), 64),
                )

                record = {
                    "symbol": symbol,
                    "target": target,
                    "model_name": model_name,
                    "feature_group": feature_group,
                    "n_splits": args.n_splits,
                    "test_size": args.test_size,
                    "min_train_size": args.min_train_size,
                    "embargo": args.embargo,
                    "val_size": args.val_size,
                    "source_best_report": str(spec_row.get("source_report_path", "")),
                    "source_best_artifact": str(spec_row.get("artifact_path", "")),
                    **asdict(spec),
                }

                print(
                    f"[{job_number}/{total_jobs}] start {symbol} {target} {model_name} "
                    f"group={feature_group}",
                    flush=True,
                )

                try:
                    result = evaluate_trial(
                        df=symbol_df,
                        feature_cols=feature_cols,
                        target_col=target,
                        spec=spec,
                        n_splits=args.n_splits,
                        test_size=args.test_size,
                        min_train_size=args.min_train_size,
                        embargo=args.embargo,
                        val_size=args.val_size,
                        device=device,
                        seed=args.seed,
                    )

                    artifact_name = (
                        f"{symbol}_{target}_{model_name}_{feature_group}_"
                        f"lag{spec.lag_window}_h{spec.hidden_size}_"
                        f"lr{spec.lr}_e{spec.epochs}_p{spec.patience}.pt"
                    ).replace("/", "_")
                    artifact_path = args.artifact_dir / target / symbol / artifact_name
                    artifact_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "symbol": symbol,
                            "model_name": model_name,
                            "feature_group": feature_group,
                            "target": target,
                            "config": asdict(spec),
                            "metrics": result["overall"],
                            "n_features": int(result["n_features"]),
                            "seq_len": int(result["seq_len"]),
                            "state_dict": result["checkpoint_state"],
                            "source_best_report": str(spec_row.get("source_report_path", "")),
                            "source_best_artifact": str(spec_row.get("artifact_path", "")),
                        },
                        artifact_path,
                    )

                    rows.append(
                        {
                            **record,
                            "status": "ok",
                            "note": "",
                            **result["overall"],
                            "n_rows": result["n_rows"],
                            "n_features": result["n_features"],
                            "seq_len": result["seq_len"],
                            "n_test_total": result["n_test_total"],
                            "avg_epochs_trained": result["avg_epochs_trained"],
                            "artifact_path": str(artifact_path),
                        }
                    )
                    print(
                        f"[{job_number}/{total_jobs}] ok {symbol} {target} {model_name}: "
                        f"rmse={result['overall']['rmse']:.6f} artifact={artifact_path}",
                        flush=True,
                    )
                except Exception as exc:
                    rows.append(
                        {
                            **record,
                            "status": "error",
                            "note": str(exc),
                            "rmse": np.nan,
                            "mae": np.nan,
                            "directional_accuracy": np.nan,
                            "sharpe": np.nan,
                            "n_rows": int(len(symbol_df)),
                            "n_features": len(feature_cols),
                            "seq_len": spec.lag_window,
                            "n_test_total": 0,
                            "avg_epochs_trained": 0,
                            "artifact_path": "",
                        }
                    )
                    print(
                        f"[{job_number}/{total_jobs}] error {symbol} {target} {model_name}: {exc}",
                        flush=True,
                    )

                write_outputs(rows=rows, args=args)

    write_outputs(rows=rows, args=args)
    print(f"Results CSV: {args.results_csv}")
    print(f"Summary JSON: {args.summary_json}")
    print(f"Summary MD: {args.summary_md}")


if __name__ == "__main__":
    main()
