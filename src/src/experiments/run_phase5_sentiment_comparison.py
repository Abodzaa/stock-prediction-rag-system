#!/usr/bin/env python3
"""Run and compare FinBERT vs RAG sentiment pipelines under identical walk-forward settings."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASE_FEATURES = Path("data/processed/features/research_features_panel.csv")
DEFAULT_NEWS_CSV = Path("data/raw_news/yfinance_sp500_news.csv")
DEFAULT_REPORT_CSV = Path("reports/sentiment_pipeline_comparison.csv")
DEFAULT_REPORT_MD = Path("reports/sentiment_pipeline_comparison.md")
DEFAULT_REPORT_JSON = Path("reports/sentiment_pipeline_comparison.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FinBERT and RAG sentiment pipelines.")
    parser.add_argument("--news-csv", type=Path, default=DEFAULT_NEWS_CSV)
    parser.add_argument("--base-features", type=Path, default=DEFAULT_BASE_FEATURES)
    parser.add_argument(
        "--runner-script",
        default="src/experiments/run_panel_walkforward_experiments.py",
        help="Experiment runner script path (panel runner recommended).",
    )
    parser.add_argument("--target", default="target_h1", choices=["target_h1", "target_h5"])
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--test-size", type=int, default=126)
    parser.add_argument("--min-train-size", type=int, default=756)
    parser.add_argument("--embargo", type=int, default=5)
    parser.add_argument("--models", default="elastic_net,ridge,hist_gbr")
    parser.add_argument("--lag-window-grid", default="10,20")
    parser.add_argument("--epochs-grid", default="150,300")
    parser.add_argument("--patience-grid", default="10,20")
    parser.add_argument("--max-train-rows", type=int, default=150000)
    parser.add_argument("--max-test-rows", type=int, default=80000)
    parser.add_argument("--finbert-max-rows", type=int, default=0)
    parser.add_argument("--rag-max-rows", type=int, default=0)
    parser.add_argument("--rag-backend", choices=["sbert", "gemini"], default="sbert")
    parser.add_argument("--rag-embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--rag-k", type=int, default=8)
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--gemini-api-key", default="")
    parser.add_argument("--gemini-temperature", type=float, default=0.0)
    parser.add_argument("--gemini-max-output-tokens", type=int, default=128)
    parser.add_argument("--gemini-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--rag-cache-csv", type=Path, default=Path("data/processed/features/news_sentiment_rag_gemini_cache.csv"))
    parser.add_argument("--sentiment-device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--skip-builders", action="store_true")
    return parser.parse_args()


def run_cmd(args: list[str]) -> None:
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(args)}")


def best_model_row(model_set_path: Path, model_name: str) -> dict | None:
    if not model_set_path.exists():
        return None

    df = pd.read_csv(model_set_path)
    df = df[(df["model"] == model_name) & (df["status"] == "ok")].copy()
    if df.empty:
        return None
    row = df.sort_values("rmse").iloc[0]
    return row.to_dict()


def extract_score(run_dir: Path, pipeline_name: str) -> dict:
    model_set = run_dir / "model_set_results.csv"
    model_family = run_dir / "model_family_results.csv"

    c = best_model_row(model_set, "Model_C_sentiment_only")
    d = best_model_row(model_set, "Model_D_price_shuffled_sentiment")
    e = best_model_row(model_set, "Model_E_price_lagged_sentiment")

    summary = {
        "pipeline": pipeline_name,
        "track": "baseline" if pipeline_name == "finbert" else "challenger",
        "model_c_rmse": c["rmse"] if c else None,
        "model_d_rmse": d["rmse"] if d else None,
        "model_e_rmse": e["rmse"] if e else None,
        "model_c_dir_acc": c["directional_accuracy"] if c else None,
        "model_c_sharpe": c["sharpe"] if c else None,
        "model_c_family": c["model_family"] if c else None,
        "model_c_note": c["note"] if c else "",
        "model_family_results": str(model_family),
        "model_set_results": str(model_set),
    }

    if c and d:
        summary["delta_c_vs_d_rmse"] = float(c["rmse"]) - float(d["rmse"])
    else:
        summary["delta_c_vs_d_rmse"] = None

    if c and e:
        summary["delta_c_vs_e_rmse"] = float(c["rmse"]) - float(e["rmse"])
    else:
        summary["delta_c_vs_e_rmse"] = None

    return summary


def build_and_run_pipeline(
    pipeline_name: str,
    args: argparse.Namespace,
    sentiment_daily_path: Path,
    sentiment_prefix: str,
) -> dict:
    run_dir = Path("reports") / f"sentiment_eval_{pipeline_name}_{args.target}"
    run_dir.mkdir(parents=True, exist_ok=True)

    merged_features = Path("data/processed/features") / f"research_features_{pipeline_name}_{args.target}.csv"

    run_cmd(
        [
            sys.executable,
            "src/nlp/merge_sentiment_features.py",
            "--base-features",
            str(args.base_features),
            "--sentiment-daily",
            str(sentiment_daily_path),
            "--output-features",
            str(merged_features),
        ]
    )

    run_cmd(
        [
            sys.executable,
            args.runner_script,
            "--features-csv",
            str(merged_features),
            "--target",
            args.target,
            "--n-splits",
            str(args.n_splits),
            "--test-size",
            str(args.test_size),
            "--min-train-size",
            str(args.min_train_size),
            "--embargo",
            str(args.embargo),
            "--models",
            args.models,
            "--lag-window-grid",
            args.lag_window_grid,
            "--epochs-grid",
            args.epochs_grid,
            "--patience-grid",
            args.patience_grid,
            "--max-train-rows",
            str(args.max_train_rows),
            "--max-test-rows",
            str(args.max_test_rows),
            "--sentiment-prefix",
            sentiment_prefix,
            "--model-family-results",
            str(run_dir / "model_family_results.csv"),
            "--group-results",
            str(run_dir / "feature_group_results.csv"),
            "--model-results",
            str(run_dir / "model_set_results.csv"),
            "--summary-md",
            str(run_dir / "summary.md"),
            "--summary-json",
            str(run_dir / "summary.json"),
            "--model-artifacts-dir",
            str(Path("artifacts/models") / f"sentiment_eval_{pipeline_name}_{args.target}"),
            "--tune",
        ]
    )

    return extract_score(run_dir=run_dir, pipeline_name=pipeline_name)


def write_reports(rows: list[dict], args: argparse.Namespace) -> None:
    df = pd.DataFrame(rows)
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(args.report_csv, index=False)

    if df.empty:
        payload = {"winner": None, "rows": []}
        args.report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        args.report_md.write_text("# Sentiment Pipeline Comparison\n\nNo rows generated.\n", encoding="utf-8")
        return

    ranked = df.copy()
    ranked["_rank_rmse"] = ranked["model_c_rmse"].fillna(1e9)
    ranked = ranked.sort_values("_rank_rmse").reset_index(drop=True)
    winner = ranked.iloc[0].to_dict()

    payload = {
        "target": args.target,
        "winner": winner,
        "recommended_baseline": "finbert",
        "recommended_challenger": "rag",
        "rows": df.to_dict(orient="records"),
        "selection_rule": "Minimum Model_C_sentiment_only RMSE; tie-break by higher directional accuracy.",
    }
    args.report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Sentiment Pipeline Comparison",
        "",
        f"- Target: {args.target}",
        "- Selection rule: minimum Model C RMSE.",
        "",
        "## Ranking",
    ]

    for _, r in ranked.iterrows():
        lines.append(
            "- "
            f"{r['pipeline']}: "
            f"C_rmse={r['model_c_rmse'] if pd.notna(r['model_c_rmse']) else 'na'}, "
            f"C_dir_acc={r['model_c_dir_acc'] if pd.notna(r['model_c_dir_acc']) else 'na'}, "
            f"C_sharpe={r['model_c_sharpe'] if pd.notna(r['model_c_sharpe']) else 'na'}, "
            f"delta(C-D)_rmse={r['delta_c_vs_d_rmse'] if pd.notna(r['delta_c_vs_d_rmse']) else 'na'}, "
            f"delta(C-E)_rmse={r['delta_c_vs_e_rmse'] if pd.notna(r['delta_c_vs_e_rmse']) else 'na'}"
        )

    lines.extend(
        [
            "",
            "## Winner",
            f"- {winner['pipeline']}",
            "",
            "## Production Tracks",
            "- baseline: finbert",
            "- challenger: rag",
            "",
            "## Interpretation",
            "- If C beats D (lower RMSE), sentiment carries non-random signal beyond shuffled control.",
            "- If C beats E, contemporaneous sentiment is more informative than a lagged variant.",
            "",
        ]
    )

    args.report_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    finbert_daily = Path("data/processed/features/news_sentiment_finbert_daily.csv")
    rag_daily = Path("data/processed/features/news_sentiment_rag_daily.csv")

    if not args.skip_builders:
        run_cmd(
            [
                sys.executable,
                "src/nlp/build_sentiment_finbert.py",
                "--input-csv",
                str(args.news_csv),
                "--daily-output-csv",
                str(finbert_daily),
                "--max-rows",
                str(args.finbert_max_rows),
                "--device",
                args.sentiment_device,
            ]
            + (["--require-cuda"] if args.require_cuda else [])
        )

        run_cmd(
            [
                sys.executable,
                "src/nlp/build_sentiment_rag.py",
                "--input-csv",
                str(args.news_csv),
                "--daily-output-csv",
                str(rag_daily),
                "--max-rows",
                str(args.rag_max_rows),
                "--backend",
                args.rag_backend,
                "--embedding-model",
                args.rag_embedding_model,
                "--k",
                str(args.rag_k),
                "--gemini-model",
                args.gemini_model,
                "--gemini-temperature",
                str(args.gemini_temperature),
                "--gemini-max-output-tokens",
                str(args.gemini_max_output_tokens),
                "--gemini-sleep-seconds",
                str(args.gemini_sleep_seconds),
                "--cache-csv",
                str(args.rag_cache_csv),
                "--device",
                args.sentiment_device,
            ]
            + (["--gemini-api-key", args.gemini_api_key] if args.gemini_api_key else [])
            + (["--require-cuda"] if args.require_cuda else [])
        )

    rows = []
    rows.append(
        build_and_run_pipeline(
            pipeline_name="finbert",
            args=args,
            sentiment_daily_path=finbert_daily,
            sentiment_prefix="g5_sent_finbert_",
        )
    )
    rows.append(
        build_and_run_pipeline(
            pipeline_name="rag",
            args=args,
            sentiment_daily_path=rag_daily,
            sentiment_prefix="g5_sent_rag_",
        )
    )

    write_reports(rows=rows, args=args)

    print(f"Comparison CSV: {args.report_csv}")
    print(f"Comparison MD: {args.report_md}")
    print(f"Comparison JSON: {args.report_json}")


if __name__ == "__main__":
    main()
