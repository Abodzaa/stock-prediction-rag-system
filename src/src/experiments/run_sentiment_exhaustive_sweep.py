#!/usr/bin/env python3
"""Run checkpointed exhaustive SBERT sentiment sweeps with resume and early-stop."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "src/experiments/run_phase5_sentiment_comparison.py"

DEFAULT_NEWS_CSV = Path("data/raw_news/yfinance_sp500_news.csv")
DEFAULT_BASE_FEATURES = Path("data/processed/features/research_features_panel.csv")
DEFAULT_RESULTS_CSV = Path("reports/sentiment_exhaustive_results.csv")
DEFAULT_CHECKPOINT_JSON = Path("reports/sentiment_exhaustive_checkpoint.json")
DEFAULT_LEADERBOARD_MD = Path("reports/sentiment_exhaustive_top5.md")
DEFAULT_LEADERBOARD_JSON = Path("reports/sentiment_exhaustive_top5.json")
DEFAULT_RUNS_DIR = Path("reports/sentiment_exhaustive_runs")


@dataclass(frozen=True)
class SweepConfig:
    target: str
    models: str
    lag_window: int
    epochs: int
    patience: int
    max_train_rows: int
    max_test_rows: int

    def config_id(self) -> str:
        model_tag = self.models.replace(",", "-")
        return (
            f"{self.target}__m_{model_tag}__lag_{self.lag_window}"
            f"__e_{self.epochs}__p_{self.patience}"
            f"__tr_{self.max_train_rows}__te_{self.max_test_rows}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpointed exhaustive sweep for SBERT sentiment comparison.")
    parser.add_argument("--news-csv", type=Path, default=DEFAULT_NEWS_CSV)
    parser.add_argument("--base-features", type=Path, default=DEFAULT_BASE_FEATURES)
    parser.add_argument("--targets", default="target_h1,target_h5")
    parser.add_argument(
        "--model-sets",
        default="elastic_net,ridge,hist_gbr;elastic_net,ridge,random_forest,hist_gbr;elastic_net,ridge,mlp,hist_gbr",
        help="Semicolon-separated model lists.",
    )
    parser.add_argument("--lag-window-grid", default="0")
    parser.add_argument("--epochs-grid", default="120,150,300,600")
    parser.add_argument("--patience-grid", default="5,10,20,40")
    parser.add_argument("--max-train-rows-grid", default="180000,220000,300000")
    parser.add_argument("--max-test-rows-grid", default="70000,90000,120000")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--test-size", type=int, default=126)
    parser.add_argument("--min-train-size", type=int, default=1260)
    parser.add_argument("--embargo", type=int, default=5)
    parser.add_argument("--rag-backend", choices=["sbert"], default="sbert")
    parser.add_argument("--sentiment-device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--skip-builders", action="store_true", default=True)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--checkpoint-json", type=Path, default=DEFAULT_CHECKPOINT_JSON)
    parser.add_argument("--leaderboard-md", type=Path, default=DEFAULT_LEADERBOARD_MD)
    parser.add_argument("--leaderboard-json", type=Path, default=DEFAULT_LEADERBOARD_JSON)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-improve", type=float, default=1e-6)
    parser.add_argument("--stop-no-improve", type=int, default=10)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one comma-separated value.")
    return vals


def parse_int_list(raw: str) -> list[int]:
    vals = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not vals:
        raise ValueError("Expected at least one integer value.")
    return vals


def parse_model_sets(raw: str) -> list[str]:
    chunks = [x.strip() for x in raw.split(";") if x.strip()]
    if not chunks:
        raise ValueError("Expected at least one model-set entry.")
    for chunk in chunks:
        _ = parse_csv_list(chunk)
    return chunks


def to_rel(path_value: Path) -> str:
    path_value = path_value.resolve()
    return str(path_value.relative_to(ROOT))


def load_checkpoint(path_value: Path, targets: list[str]) -> dict:
    if not path_value.exists():
        return {
            "completed_ids": [],
            "best_rmse_by_target": {t: None for t in targets},
            "no_improve_streak_by_target": {t: 0 for t in targets},
            "stopped_targets": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": None,
        }

    payload = json.loads(path_value.read_text(encoding="utf-8"))
    payload.setdefault("completed_ids", [])
    payload.setdefault("best_rmse_by_target", {})
    payload.setdefault("no_improve_streak_by_target", {})
    payload.setdefault("stopped_targets", [])

    for t in targets:
        payload["best_rmse_by_target"].setdefault(t, None)
        payload["no_improve_streak_by_target"].setdefault(t, 0)

    return payload


def write_checkpoint(path_value: Path, payload: dict) -> None:
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    path_value.parent.mkdir(parents=True, exist_ok=True)
    path_value.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_best_row(comparison_csv: Path) -> dict | None:
    if not comparison_csv.exists():
        return None

    df = pd.read_csv(comparison_csv)
    if df.empty:
        return None

    df = df.copy()
    df["model_c_rmse"] = pd.to_numeric(df["model_c_rmse"], errors="coerce")
    df["model_c_dir_acc"] = pd.to_numeric(df["model_c_dir_acc"], errors="coerce")

    if df["model_c_rmse"].notna().sum() == 0:
        return None

    df = df.sort_values(["model_c_rmse", "model_c_dir_acc"], ascending=[True, False])
    return df.iloc[0].to_dict()


def write_leaderboards(results_df: pd.DataFrame, top_k: int, md_out: Path, json_out: Path) -> None:
    ok = results_df[results_df["status"] == "ok"].copy()
    ok["best_model_c_rmse"] = pd.to_numeric(ok["best_model_c_rmse"], errors="coerce")
    ok["best_model_c_dir_acc"] = pd.to_numeric(ok["best_model_c_dir_acc"], errors="coerce")
    ok = ok.dropna(subset=["best_model_c_rmse"])

    leaderboard = {}
    lines = [
        "# Sentiment Exhaustive Sweep Top Results",
        "",
        "Selection rule: lowest best_model_c_rmse, tie-break by higher best_model_c_dir_acc.",
        "",
    ]

    for target in sorted(ok["target"].unique()):
        part = ok[ok["target"] == target].copy()
        part = part.sort_values(["best_model_c_rmse", "best_model_c_dir_acc"], ascending=[True, False]).head(top_k)
        leaderboard[target] = part.to_dict(orient="records")

        lines.append(f"## {target}")
        if part.empty:
            lines.append("- none")
            lines.append("")
            continue

        for _, row in part.iterrows():
            lines.append(
                "- "
                f"rmse={row['best_model_c_rmse']:.9f}, dir_acc={row['best_model_c_dir_acc']:.6f}, "
                f"pipeline={row['best_pipeline']}, models={row['models']}, lag={int(row['lag_window'])}, "
                f"epochs={int(row['epochs'])}, patience={int(row['patience'])}, "
                f"train_rows={int(row['max_train_rows'])}, test_rows={int(row['max_test_rows'])}, "
                f"run_id={row['run_id']}"
            )
        lines.append("")

    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(lines), encoding="utf-8")
    json_out.write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")


def run_once(args: argparse.Namespace, cfg: SweepConfig, run_dir: Path, skip_builders: bool) -> tuple[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    report_csv = run_dir / "comparison.csv"
    report_md = run_dir / "comparison.md"
    report_json = run_dir / "comparison.json"

    cmd = [
        sys.executable,
        str(RUNNER),
        "--news-csv",
        to_rel(args.news_csv),
        "--base-features",
        to_rel(args.base_features),
        "--runner-script",
        "src/experiments/run_panel_walkforward_experiments.py",
        "--target",
        cfg.target,
        "--n-splits",
        str(args.n_splits),
        "--test-size",
        str(args.test_size),
        "--min-train-size",
        str(args.min_train_size),
        "--embargo",
        str(args.embargo),
        "--models",
        cfg.models,
        "--lag-window-grid",
        str(cfg.lag_window),
        "--epochs-grid",
        str(cfg.epochs),
        "--patience-grid",
        str(cfg.patience),
        "--max-train-rows",
        str(cfg.max_train_rows),
        "--max-test-rows",
        str(cfg.max_test_rows),
        "--rag-backend",
        args.rag_backend,
        "--sentiment-device",
        args.sentiment_device,
        "--report-csv",
        str(report_csv.relative_to(ROOT)),
        "--report-md",
        str(report_md.relative_to(ROOT)),
        "--report-json",
        str(report_json.relative_to(ROOT)),
    ]

    if args.require_cuda:
        cmd.append("--require-cuda")
    if skip_builders:
        cmd.append("--skip-builders")

    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if proc.returncode != 0:
        details = "\n".join(x for x in [stdout, stderr] if x)
        return "error", details[:6000]

    return "ok", stdout[-2000:] if stdout else ""


def main() -> None:
    args = parse_args()

    args.news_csv = (ROOT / args.news_csv).resolve()
    args.base_features = (ROOT / args.base_features).resolve()

    if not args.news_csv.exists():
        raise FileNotFoundError(f"News CSV not found: {args.news_csv}")
    if not args.base_features.exists():
        raise FileNotFoundError(f"Base features CSV not found: {args.base_features}")
    if not RUNNER.exists():
        raise FileNotFoundError(f"Runner not found: {RUNNER}")

    targets = parse_csv_list(args.targets)
    model_sets = parse_model_sets(args.model_sets)
    lag_grid = parse_int_list(args.lag_window_grid)
    epochs_grid = parse_int_list(args.epochs_grid)
    patience_grid = parse_int_list(args.patience_grid)
    train_grid = parse_int_list(args.max_train_rows_grid)
    test_grid = parse_int_list(args.max_test_rows_grid)

    configs = [
        SweepConfig(
            target=t,
            models=m,
            lag_window=lag,
            epochs=e,
            patience=p,
            max_train_rows=tr,
            max_test_rows=te,
        )
        for t, m, lag, e, p, tr, te in product(targets, model_sets, lag_grid, epochs_grid, patience_grid, train_grid, test_grid)
    ]

    checkpoint_path = (ROOT / args.checkpoint_json).resolve()
    results_path = (ROOT / args.results_csv).resolve()
    md_path = (ROOT / args.leaderboard_md).resolve()
    json_path = (ROOT / args.leaderboard_json).resolve()
    runs_dir = (ROOT / args.runs_dir).resolve()

    state = load_checkpoint(checkpoint_path, targets=targets)
    completed_ids = set(state["completed_ids"])
    stopped_targets = set(state.get("stopped_targets", []))

    if results_path.exists():
        results_df = pd.read_csv(results_path)
        rows = results_df.to_dict(orient="records")
    else:
        rows = []

    executed = 0

    for cfg in configs:
        if cfg.target in stopped_targets:
            continue

        run_id = cfg.config_id()
        if run_id in completed_ids:
            continue

        if args.max_runs > 0 and executed >= args.max_runs:
            break

        run_dir = runs_dir / cfg.target / run_id
        skip_builders = bool(args.skip_builders)

        status, note = run_once(args=args, cfg=cfg, run_dir=run_dir, skip_builders=skip_builders)

        best_row = read_best_row(run_dir / "comparison.csv") if status == "ok" else None
        best_rmse = float(best_row["model_c_rmse"]) if best_row and pd.notna(best_row.get("model_c_rmse")) else None
        best_dir = float(best_row["model_c_dir_acc"]) if best_row and pd.notna(best_row.get("model_c_dir_acc")) else None
        best_pipeline = str(best_row["pipeline"]) if best_row and best_row.get("pipeline") else ""

        row = {
            "run_id": run_id,
            "status": status,
            "target": cfg.target,
            "models": cfg.models,
            "lag_window": cfg.lag_window,
            "epochs": cfg.epochs,
            "patience": cfg.patience,
            "max_train_rows": cfg.max_train_rows,
            "max_test_rows": cfg.max_test_rows,
            "best_pipeline": best_pipeline,
            "best_model_c_rmse": best_rmse,
            "best_model_c_dir_acc": best_dir,
            "run_report_csv": str((run_dir / "comparison.csv").relative_to(ROOT)),
            "run_report_md": str((run_dir / "comparison.md").relative_to(ROOT)),
            "run_report_json": str((run_dir / "comparison.json").relative_to(ROOT)),
            "note": note,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        rows.append(row)

        completed_ids.add(run_id)
        state["completed_ids"] = sorted(completed_ids)

        if status == "ok" and best_rmse is not None:
            prev_best = state["best_rmse_by_target"].get(cfg.target)
            if prev_best is None or (prev_best - best_rmse) >= args.min_improve:
                state["best_rmse_by_target"][cfg.target] = best_rmse
                state["no_improve_streak_by_target"][cfg.target] = 0
            else:
                state["no_improve_streak_by_target"][cfg.target] = int(state["no_improve_streak_by_target"].get(cfg.target, 0)) + 1
        else:
            state["no_improve_streak_by_target"][cfg.target] = int(state["no_improve_streak_by_target"].get(cfg.target, 0)) + 1

        if int(state["no_improve_streak_by_target"].get(cfg.target, 0)) >= int(args.stop_no_improve):
            stopped_targets.add(cfg.target)
            state["stopped_targets"] = sorted(stopped_targets)

        results_df = pd.DataFrame(rows)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)

        write_checkpoint(checkpoint_path, state)
        write_leaderboards(results_df=results_df, top_k=args.top_k, md_out=md_path, json_out=json_path)

        executed += 1

        if status != "ok" and not args.continue_on_error:
            break

    print("Sentiment exhaustive sweep complete.")
    print(f"Executed runs this invocation: {executed}")
    print(f"Results CSV: {results_path.relative_to(ROOT)}")
    print(f"Checkpoint JSON: {checkpoint_path.relative_to(ROOT)}")
    print(f"Leaderboard MD: {md_path.relative_to(ROOT)}")
    print(f"Leaderboard JSON: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
