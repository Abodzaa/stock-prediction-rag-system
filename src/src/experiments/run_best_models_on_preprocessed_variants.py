#!/usr/bin/env python3
"""Run best h1/h5 model settings on each prepared preprocessing variant."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "src/experiments/run_panel_walkforward_experiments.py"

VARIANTS = [
    ("v1_winsor_robustz", "data/processed/features/research_features_panel_pre_v1.csv"),
    ("v2_ema_standardz", "data/processed/features/research_features_panel_pre_v2.csv"),
    ("v3_rollmed_minmax", "data/processed/features/research_features_panel_pre_v3.csv"),
    ("v4_hampel_ranknorm", "data/processed/features/research_features_panel_pre_v4.csv"),
    ("v5_iqrclip_log_cs_z", "data/processed/features/research_features_panel_pre_v5.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run best configs on preprocessing variants.")
    parser.add_argument("--models", default="elastic_net,ridge,hist_gbr")
    parser.add_argument("--lag-window", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--test-size", type=int, default=126)
    parser.add_argument("--min-train-size", type=int, default=1260)
    parser.add_argument("--embargo", type=int, default=5)
    parser.add_argument("--max-train-rows", type=int, default=260000)
    parser.add_argument("--max-test-rows", type=int, default=100000)
    parser.add_argument("--summary-csv", type=Path, default=Path("reports/preprocessing_variants_best_models_summary.csv"))
    parser.add_argument("--summary-md", type=Path, default=Path("reports/preprocessing_variants_best_models_summary.md"))
    parser.add_argument("--summary-json", type=Path, default=Path("reports/preprocessing_variants_best_models_summary.json"))
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = "\n".join(x for x in [proc.stdout, proc.stderr] if x)
        raise RuntimeError(msg[:6000])


def best_row(model_family_csv: Path) -> dict | None:
    if not model_family_csv.exists():
        return None
    df = pd.read_csv(model_family_csv)
    part = df[df["status"] == "ok"].copy()
    if part.empty:
        return None
    row = part.sort_values("rmse").iloc[0]
    return row.to_dict()


def write_reports(rows: list[dict], args: argparse.Namespace) -> None:
    df = pd.DataFrame(rows)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(args.summary_csv, index=False)
    args.summary_json.write_text(json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8")

    lines = [
        "# Best Models on Preprocessing Variants",
        "",
        "Settings: models=elastic_net,ridge,hist_gbr | lag=0 | epochs=120 | patience=5",
        "",
    ]
    if df.empty:
        lines.append("No rows produced.")
    else:
        for target in ["target_h1", "target_h5"]:
            lines.append(f"## {target}")
            part = df[df["target"] == target].copy().sort_values("rmse")
            if part.empty:
                lines.append("- none")
            else:
                for _, r in part.iterrows():
                    lines.append(
                        "- "
                        f"{r['variant']}: rmse={r['rmse']:.9f}, dir_acc={r['directional_accuracy']:.6f}, "
                        f"model_family={r['model_family']}, group={r['feature_group']}"
                    )
            lines.append("")

    args.summary_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    rows: list[dict] = []
    errors: list[dict] = []

    for variant, features_rel in VARIANTS:
        features_csv = ROOT / features_rel
        if not features_csv.exists():
            msg = f"Missing variant CSV: {features_csv}"
            if not args.continue_on_error:
                raise FileNotFoundError(msg)
            errors.append({"variant": variant, "target": "all", "error": msg})
            continue

        for target in ["target_h1", "target_h5"]:
            out_dir = ROOT / "reports" / "preprocessing_variants" / variant / target
            out_dir.mkdir(parents=True, exist_ok=True)

            model_family_csv = out_dir / "model_family_results.csv"
            group_csv = out_dir / "feature_group_results.csv"
            model_set_csv = out_dir / "model_set_results.csv"
            summary_md = out_dir / "summary.md"
            summary_json = out_dir / "summary.json"

            cmd = [
                sys.executable,
                str(RUNNER),
                "--features-csv",
                str(features_csv.relative_to(ROOT)),
                "--target",
                target,
                "--models",
                args.models,
                "--lag-window",
                str(args.lag_window),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--n-splits",
                str(args.n_splits),
                "--test-size",
                str(args.test_size),
                "--min-train-size",
                str(args.min_train_size),
                "--embargo",
                str(args.embargo),
                "--max-train-rows",
                str(args.max_train_rows),
                "--max-test-rows",
                str(args.max_test_rows),
                "--model-family-results",
                str(model_family_csv.relative_to(ROOT)),
                "--group-results",
                str(group_csv.relative_to(ROOT)),
                "--model-results",
                str(model_set_csv.relative_to(ROOT)),
                "--summary-md",
                str(summary_md.relative_to(ROOT)),
                "--summary-json",
                str(summary_json.relative_to(ROOT)),
                "--model-artifacts-dir",
                str((ROOT / "artifacts/models/preprocessing_variants" / variant / target).relative_to(ROOT)),
            ]

            try:
                run_cmd(cmd)
                best = best_row(model_family_csv)
                if best is None:
                    rows.append(
                        {
                            "variant": variant,
                            "target": target,
                            "status": "no_successful_rows",
                            "feature_group": "",
                            "model_family": "",
                            "rmse": None,
                            "directional_accuracy": None,
                            "sharpe": None,
                            "model_family_results": str(model_family_csv.relative_to(ROOT)),
                        }
                    )
                else:
                    rows.append(
                        {
                            "variant": variant,
                            "target": target,
                            "status": "ok",
                            "feature_group": best.get("feature_group", ""),
                            "model_family": best.get("model_family", ""),
                            "rmse": float(best.get("rmse")),
                            "directional_accuracy": float(best.get("directional_accuracy")),
                            "sharpe": float(best.get("sharpe")),
                            "model_family_results": str(model_family_csv.relative_to(ROOT)),
                        }
                    )
            except Exception as exc:
                err = {"variant": variant, "target": target, "error": str(exc)}
                errors.append(err)
                if not args.continue_on_error:
                    raise

    if errors:
        err_df = pd.DataFrame(errors)
        err_csv = ROOT / "reports/preprocessing_variants_best_models_errors.csv"
        err_csv.parent.mkdir(parents=True, exist_ok=True)
        err_df.to_csv(err_csv, index=False)

    write_reports(rows=rows, args=args)

    print(f"Summary CSV: {args.summary_csv}")
    print(f"Summary MD: {args.summary_md}")
    print(f"Summary JSON: {args.summary_json}")
    if errors:
        print("Some runs had errors. See reports/preprocessing_variants_best_models_errors.csv")


if __name__ == "__main__":
    main()
