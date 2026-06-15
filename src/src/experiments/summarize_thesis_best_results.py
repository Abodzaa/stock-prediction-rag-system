#!/usr/bin/env python3
"""Build thesis-ready best-result tables from existing report artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SUMMARY_CSV = Path("reports/thesis_best_model_summary.csv")
DEFAULT_CANDIDATES_CSV = Path("reports/thesis_best_model_candidates.csv")

RANK_COLUMNS = ["rmse", "directional_accuracy", "sharpe"]

CURATED_DEEP_REPORTS = [
    {
        "path": Path("reports/deep_torch_model_results_h1_best_configs.csv"),
        "summary_path": Path("reports/deep_torch_model_summary_h1.json"),
        "experiment_scope": "deep_daily_price",
        "target_scope": "h1",
        "dataset_scope": "sp500_index_daily",
        "source_label": "deep_torch_h1_best_configs",
        "preprocessing_method": "daily_features_train_fold_standardization",
        "preprocessing_algorithm": (
            "Daily S&P 500 index feature matrix; model runner standardizes sequence features "
            "within each train fold only."
        ),
    },
    {
        "path": Path("reports/deep_torch_model_results_h5_best_configs.csv"),
        "summary_path": Path("reports/deep_torch_model_summary_h5.json"),
        "experiment_scope": "deep_daily_price",
        "target_scope": "h5",
        "dataset_scope": "sp500_index_daily",
        "source_label": "deep_torch_h5_best_configs",
        "preprocessing_method": "daily_features_train_fold_standardization",
        "preprocessing_algorithm": (
            "Daily S&P 500 index feature matrix; model runner standardizes sequence features "
            "within each train fold only."
        ),
    },
    {
        "path": Path("reports/deep_preprocessed_sentiment_h1_results_best_configs.csv"),
        "summary_path": Path("reports/deep_preprocessed_sentiment_h1_summary.json"),
        "experiment_scope": "deep_daily_sentiment",
        "target_scope": "h1_sentiment",
        "dataset_scope": "sp500_index_daily_sentiment",
        "source_label": "deep_preprocessed_sentiment_h1_best_configs",
        "preprocessing_method": "daily_preprocessed_finbert_sentiment",
        "preprocessing_algorithm": (
            "Daily price/technical/breadth features merged with FinBERT daily sentiment; "
            "deep sequence features are standardized within each train fold only."
        ),
    },
    {
        "path": Path("reports/deep_preprocessed_sentiment_h5_results_best_configs.csv"),
        "summary_path": Path("reports/deep_preprocessed_sentiment_h5_summary.json"),
        "experiment_scope": "deep_daily_sentiment",
        "target_scope": "h5_sentiment",
        "dataset_scope": "sp500_index_daily_sentiment",
        "source_label": "deep_preprocessed_sentiment_h5_best_configs",
        "preprocessing_method": "daily_preprocessed_finbert_sentiment",
        "preprocessing_algorithm": (
            "Daily price/technical/breadth features merged with FinBERT daily sentiment; "
            "deep sequence features are standardized within each train fold only."
        ),
    },
    {
        "path": Path("reports/deep_preprocessed_sentiment_h1_lstm_targeted_tune_results_best_configs.csv"),
        "summary_path": Path("reports/deep_preprocessed_sentiment_h1_lstm_targeted_tune_summary.json"),
        "experiment_scope": "deep_daily_sentiment",
        "target_scope": "h1_sentiment",
        "dataset_scope": "sp500_index_daily_sentiment",
        "source_label": "deep_preprocessed_sentiment_h1_lstm_targeted_tune",
        "preprocessing_method": "daily_preprocessed_finbert_sentiment",
        "preprocessing_algorithm": (
            "Daily price/technical/breadth features merged with FinBERT daily sentiment; "
            "targeted LSTM tune; sequence features are standardized within each train fold only."
        ),
    },
    {
        "path": Path("reports/deep_preprocessed_sentiment_h1_gru_targeted_tune_results_best_configs.csv"),
        "summary_path": Path("reports/deep_preprocessed_sentiment_h1_gru_targeted_tune_summary.json"),
        "experiment_scope": "deep_daily_sentiment",
        "target_scope": "h1_sentiment",
        "dataset_scope": "sp500_index_daily_sentiment",
        "source_label": "deep_preprocessed_sentiment_h1_gru_targeted_tune",
        "preprocessing_method": "daily_preprocessed_finbert_sentiment",
        "preprocessing_algorithm": (
            "Daily price/technical/breadth features merged with FinBERT daily sentiment; "
            "targeted GRU tune; sequence features are standardized within each train fold only."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize thesis-ready best model report rows.")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--candidates-csv", type=Path, default=DEFAULT_CANDIDATES_CSV)
    return parser.parse_args()


def rel(path_value: Path | str) -> str:
    path = Path(str(path_value))
    if path.is_absolute():
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)
    return str(path)


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    return val


def maybe_int(value: Any) -> int | None:
    val = maybe_float(value)
    if val is None:
        return None
    return int(val)


def maybe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def load_json(path_value: Path) -> dict[str, Any]:
    full = ROOT / path_value
    if not full.exists():
        return {}
    return json.loads(full.read_text(encoding="utf-8"))


def preprocessing_from_metadata(metadata_path: Path, fallback_method: str) -> tuple[str, str, str]:
    meta = load_json(metadata_path)
    if not meta:
        return fallback_method, "", ""
    preprocessing = meta.get("preprocessing", {})
    denoising = preprocessing.get("denoising", [])
    normalization = preprocessing.get("normalization", [])
    steps = []
    if denoising:
        steps.append("Denoising: " + "; ".join(map(str, denoising)))
    if normalization:
        steps.append("Normalization: " + "; ".join(map(str, normalization)))
    return (
        str(meta.get("method", fallback_method)),
        " | ".join(steps),
        str(meta.get("output_csv", "")),
    )


def base_candidate(**kwargs: Any) -> dict[str, Any]:
    row = {
        "selection_view": "",
        "target": kwargs.get("target", ""),
        "target_scope": kwargs.get("target_scope", ""),
        "experiment_scope": kwargs.get("experiment_scope", ""),
        "dataset_scope": kwargs.get("dataset_scope", ""),
        "source_label": kwargs.get("source_label", ""),
        "model_name": kwargs.get("model_name", ""),
        "model_family": kwargs.get("model_family", ""),
        "feature_group": kwargs.get("feature_group", ""),
        "pipeline": kwargs.get("pipeline", ""),
        "variant": kwargs.get("variant", ""),
        "control_type": kwargs.get("control_type", ""),
        "is_control": kwargs.get("is_control", False),
        "preprocessing_method": kwargs.get("preprocessing_method", ""),
        "preprocessing_algorithm": kwargs.get("preprocessing_algorithm", ""),
        "features_csv": kwargs.get("features_csv", ""),
        "lag_window": kwargs.get("lag_window"),
        "hidden_size": kwargs.get("hidden_size"),
        "lr": kwargs.get("lr"),
        "epochs": kwargs.get("epochs"),
        "patience": kwargs.get("patience"),
        "batch_size": kwargs.get("batch_size"),
        "rmse": kwargs.get("rmse"),
        "mae": kwargs.get("mae"),
        "directional_accuracy": kwargs.get("directional_accuracy"),
        "sharpe": kwargs.get("sharpe"),
        "n_rows": kwargs.get("n_rows"),
        "n_features": kwargs.get("n_features"),
        "n_test_total": kwargs.get("n_test_total"),
        "artifact_path": kwargs.get("artifact_path", ""),
        "source_report_path": kwargs.get("source_report_path", ""),
        "source_summary_path": kwargs.get("source_summary_path", ""),
        "selection_rule": "Lowest RMSE, tie-break by higher directional accuracy then higher Sharpe.",
        "note": kwargs.get("note", ""),
    }
    return row


def append_deep_report(candidates: list[dict[str, Any]], spec: dict[str, Any]) -> None:
    full = ROOT / spec["path"]
    if not full.exists():
        return

    df = pd.read_csv(full)
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"].copy()

    for _, r in df.iterrows():
        candidates.append(
            base_candidate(
                target=str(r.get("target", "")),
                target_scope=spec["target_scope"],
                experiment_scope=spec["experiment_scope"],
                dataset_scope=spec["dataset_scope"],
                source_label=spec["source_label"],
                model_name=str(r.get("model_name", "")),
                model_family=str(r.get("model_name", "")),
                feature_group=str(r.get("feature_group", "")),
                preprocessing_method=spec["preprocessing_method"],
                preprocessing_algorithm=spec["preprocessing_algorithm"],
                lag_window=maybe_int(r.get("lag_window")),
                hidden_size=maybe_int(r.get("hidden_size")),
                lr=maybe_float(r.get("lr")),
                epochs=maybe_int(r.get("epochs")),
                patience=maybe_int(r.get("patience")),
                batch_size=maybe_int(r.get("batch_size")),
                rmse=maybe_float(r.get("rmse")),
                mae=maybe_float(r.get("mae")),
                directional_accuracy=maybe_float(r.get("directional_accuracy")),
                sharpe=maybe_float(r.get("sharpe")),
                n_rows=maybe_int(r.get("n_rows")),
                n_features=maybe_int(r.get("n_features")),
                n_test_total=maybe_int(r.get("n_test_total")),
                artifact_path=rel(r.get("artifact_path", "")),
                source_report_path=rel(spec["path"]),
                source_summary_path=rel(spec.get("summary_path", "")),
                note=maybe_str(r.get("note", "")),
            )
        )


def append_preprocessing_variants(candidates: list[dict[str, Any]]) -> None:
    summary_path = ROOT / "reports/preprocessing_variants_best_models_summary.csv"
    if not summary_path.exists():
        return

    summary = pd.read_csv(summary_path)
    seen_files = sorted({str(x) for x in summary["model_family_results"].dropna().unique()})

    for report in seen_files:
        full = ROOT / report
        if not full.exists():
            continue

        match = re.search(r"preprocessing_variants[\\/](?P<variant>[^\\/]+)[\\/](?P<target>target_h[15])", report)
        variant = match.group("variant") if match else ""
        target = match.group("target") if match else ""
        target_scope = "h1" if target == "target_h1" else "h5"
        version_match = re.search(r"(?:^|_)v(?P<version>[1-5])", variant)
        metadata_path = (
            Path(f"data/metadata/research_features_panel_pre_v{version_match.group('version')}.json")
            if version_match
            else Path("data/metadata/research_features_panel_metadata.json")
        )
        method, algorithm, features_csv = preprocessing_from_metadata(metadata_path, fallback_method=variant)

        df = pd.read_csv(full)
        if "status" in df.columns:
            df = df[df["status"].astype(str).str.lower() == "ok"].copy()

        for _, r in df.iterrows():
            candidates.append(
                base_candidate(
                    target=target,
                    target_scope=target_scope,
                    experiment_scope="panel_preprocessing",
                    dataset_scope="sp500_constituent_panel",
                    source_label="preprocessing_variants_best_models",
                    model_name=str(r.get("model_family", "")),
                    model_family=str(r.get("model_family", "")),
                    feature_group=str(r.get("feature_group", "")),
                    variant=variant,
                    preprocessing_method=method,
                    preprocessing_algorithm=algorithm,
                    features_csv=rel(features_csv),
                    lag_window=maybe_int(r.get("lag_window")),
                    epochs=maybe_int(r.get("epochs")),
                    patience=maybe_int(r.get("patience")),
                    rmse=maybe_float(r.get("rmse")),
                    mae=maybe_float(r.get("mae")),
                    directional_accuracy=maybe_float(r.get("directional_accuracy")),
                    sharpe=maybe_float(r.get("sharpe")),
                    n_rows=maybe_int(r.get("n_rows")),
                    n_features=maybe_int(r.get("n_features")),
                    n_test_total=maybe_int(r.get("n_test_total")),
                    artifact_path=rel(r.get("artifact_path", "")),
                    source_report_path=rel(report),
                    source_summary_path=rel(Path(report).with_name("summary.json")),
                )
            )


def infer_control_type(model_name: str, feature_group: str) -> tuple[bool, str]:
    text = f"{model_name} {feature_group}".lower()
    if "shuffled" in text:
        return True, "shuffled_sentiment_control"
    if "lagged" in text:
        return True, "lagged_sentiment_control"
    return False, ""


def append_sentiment_panel(candidates: list[dict[str, Any]]) -> None:
    for pipeline in ["finbert", "rag"]:
        for target in ["target_h1", "target_h5"]:
            model_set = ROOT / f"reports/sentiment_eval_{pipeline}_{target}/model_set_results.csv"
            if not model_set.exists():
                continue

            df = pd.read_csv(model_set)
            if "status" in df.columns:
                df = df[df["status"].astype(str).str.lower() == "ok"].copy()

            method = f"panel_{pipeline}_sentiment_merge"
            algorithm = (
                "Base S&P 500 constituent panel preprocessing plus daily sentiment merge; "
                f"{pipeline.upper()} sentiment features use g5 sentiment columns; "
                "sklearn pipelines impute missing values and apply RobustScaler/StandardScaler where model-specific."
            )
            target_scope = "h1_sentiment" if target == "target_h1" else "h5_sentiment"

            for _, r in df.iterrows():
                model_name = str(r.get("model", ""))
                feature_group = {
                    "Model_A_price_only": "G1_price_only",
                    "Model_B_price_full_available": "G3_plus_panel_breadth",
                    "Model_C_sentiment_only": "G5_sentiment_only",
                    "Model_D_price_shuffled_sentiment": "G3_plus_shuffled_sentiment",
                    "Model_E_price_lagged_sentiment": "G3_plus_lagged_sentiment",
                }.get(model_name, model_name)
                is_control, control_type = infer_control_type(model_name, feature_group)
                candidates.append(
                    base_candidate(
                        target=target,
                        target_scope=target_scope,
                        experiment_scope="panel_sentiment_pipeline",
                        dataset_scope="sp500_constituent_panel_sentiment",
                        source_label=f"sentiment_eval_{pipeline}_{target}",
                        model_name=model_name,
                        model_family=str(r.get("model_family", "")),
                        feature_group=feature_group,
                        pipeline=pipeline,
                        control_type=control_type,
                        is_control=is_control,
                        preprocessing_method=method,
                        preprocessing_algorithm=algorithm,
                        features_csv=rel(f"data/processed/features/research_features_{pipeline}_{target}.csv"),
                        lag_window=maybe_int(r.get("lag_window")),
                        epochs=maybe_int(r.get("epochs")),
                        patience=maybe_int(r.get("patience")),
                        rmse=maybe_float(r.get("rmse")),
                        mae=maybe_float(r.get("mae")),
                        directional_accuracy=maybe_float(r.get("directional_accuracy")),
                        sharpe=maybe_float(r.get("sharpe")),
                        n_rows=maybe_int(r.get("n_rows")),
                        n_features=maybe_int(r.get("n_features")),
                        n_test_total=maybe_int(r.get("n_test_total")),
                        artifact_path=rel(r.get("artifact_path", "")),
                        source_report_path=rel(model_set),
                        source_summary_path=rel(model_set.with_name("summary.json")),
                    note=maybe_str(r.get("note", "")),
                )
            )


def rank_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in RANK_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["rmse"])
    return out.sort_values(["rmse", "directional_accuracy", "sharpe"], ascending=[True, False, False])


def build_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    eligible = candidates[candidates["is_control"].astype(str).str.lower().isin(["false", "0"])].copy()
    ranked = rank_rows(eligible)

    summary_rows = []
    group_cols = ["target_scope", "experiment_scope"]
    for _, part in ranked.groupby(group_cols, sort=True):
        best = part.iloc[0].copy()
        best["selection_view"] = "global_best"
        summary_rows.append(best)

    per_model_cols = ["target_scope", "experiment_scope", "model_name"]
    for _, part in ranked.groupby(per_model_cols, sort=True):
        best = part.iloc[0].copy()
        best["selection_view"] = "best_per_model"
        summary_rows.append(best)

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return summary

    return rank_rows(summary).sort_values(
        ["target_scope", "experiment_scope", "selection_view", "rmse", "directional_accuracy", "sharpe"],
        ascending=[True, True, True, True, False, False],
    )


def main() -> None:
    args = parse_args()

    candidates: list[dict[str, Any]] = []
    for spec in CURATED_DEEP_REPORTS:
        append_deep_report(candidates, spec)
    append_preprocessing_variants(candidates)
    append_sentiment_panel(candidates)

    candidates_df = pd.DataFrame(candidates)
    if not candidates_df.empty:
        candidates_df = candidates_df.sort_values(
            ["target_scope", "experiment_scope", "rmse", "directional_accuracy", "sharpe"],
            ascending=[True, True, True, False, False],
        )

    summary_df = build_summary(candidates_df) if not candidates_df.empty else pd.DataFrame()

    summary_out = ROOT / args.summary_csv
    candidates_out = ROOT / args.candidates_csv
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    candidates_out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_out, index=False)
    candidates_df.to_csv(candidates_out, index=False)

    print(f"Summary CSV: {args.summary_csv}")
    print(f"Candidate CSV: {args.candidates_csv}")
    print(f"Summary rows: {len(summary_df)} | Candidate rows: {len(candidates_df)}")


if __name__ == "__main__":
    main()
