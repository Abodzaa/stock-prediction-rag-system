#!/usr/bin/env python3
"""Validate per-ticker S&P 500 OHLCV files and emit QA reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_MAPPING_CSV = Path("data/metadata/sp500_constituent_file_map.csv")
DEFAULT_CALENDAR_CSV = Path("data/raw_market/sp500_2006_2026.csv")
DEFAULT_OUTPUT_CSV = Path("data/metadata/sp500_constituent_qa_report.csv")
DEFAULT_SUMMARY_JSON = Path("data/metadata/sp500_constituent_qa_summary.json")
DEFAULT_SUMMARY_MD = Path("reports/sp500_constituent_qa_summary.md")
REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate S&P 500 per-ticker CSV data quality.")
    parser.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAPPING_CSV)
    parser.add_argument("--calendar-csv", type=Path, default=DEFAULT_CALENDAR_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument(
        "--max-missing-trading-days-warning",
        type=int,
        default=3,
        help="Warn when missing expected trading days exceed this value.",
    )
    return parser.parse_args()


def load_trading_calendar(calendar_csv: Path) -> pd.DatetimeIndex:
    if not calendar_csv.exists():
        raise FileNotFoundError(f"Calendar CSV not found: {calendar_csv}")

    cal_df = pd.read_csv(calendar_csv)
    if "Date" not in cal_df.columns:
        raise ValueError(f"Calendar CSV missing Date column: {calendar_csv}")

    dates = pd.to_datetime(cal_df["Date"], errors="coerce").dropna()
    dates = pd.DatetimeIndex(dates).sort_values().unique()
    if len(dates) == 0:
        raise ValueError("Calendar contains no valid dates.")
    return dates


def validate_one_file(
    symbol: str,
    file_path: str,
    trading_calendar: pd.DatetimeIndex,
    max_missing_trading_days_warning: int,
) -> dict:
    result = {
        "symbol": symbol,
        "file_path": file_path,
        "exists": False,
        "read_ok": False,
        "required_columns_ok": False,
        "row_count": 0,
        "date_min": "",
        "date_max": "",
        "invalid_date_count": 0,
        "duplicate_date_count": 0,
        "date_monotonic_increasing": False,
        "missing_open_count": 0,
        "missing_high_count": 0,
        "missing_low_count": 0,
        "missing_close_count": 0,
        "missing_volume_count": 0,
        "nonpositive_price_count": 0,
        "negative_volume_count": 0,
        "missing_trading_days_count": 0,
        "status": "error",
        "issues": "",
    }

    issues: list[str] = []
    path = Path(file_path)

    if not path.exists():
        issues.append("missing_file")
        result["issues"] = ";".join(issues)
        return result

    result["exists"] = True

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        issues.append(f"read_error:{exc}")
        result["issues"] = ";".join(issues)
        return result

    result["read_ok"] = True

    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_required:
        issues.append(f"missing_columns:{','.join(missing_required)}")
        result["issues"] = ";".join(issues)
        return result

    result["required_columns_ok"] = True
    result["row_count"] = int(len(df))

    parsed_dates = pd.to_datetime(df["Date"], errors="coerce")
    result["invalid_date_count"] = int(parsed_dates.isna().sum())
    if result["invalid_date_count"] > 0:
        issues.append("invalid_dates")

    work_df = df.copy()
    work_df["Date"] = parsed_dates
    work_df = work_df.dropna(subset=["Date"]).sort_values("Date")

    if len(work_df) == 0:
        issues.append("no_valid_rows")
        result["issues"] = ";".join(issues)
        return result

    result["date_min"] = work_df["Date"].iloc[0].strftime("%Y-%m-%d")
    result["date_max"] = work_df["Date"].iloc[-1].strftime("%Y-%m-%d")
    result["duplicate_date_count"] = int(work_df["Date"].duplicated().sum())
    result["date_monotonic_increasing"] = bool(work_df["Date"].is_monotonic_increasing)

    if result["duplicate_date_count"] > 0:
        issues.append("duplicate_dates")
    if not result["date_monotonic_increasing"]:
        issues.append("non_monotonic_dates")

    result["missing_open_count"] = int(work_df["Open"].isna().sum())
    result["missing_high_count"] = int(work_df["High"].isna().sum())
    result["missing_low_count"] = int(work_df["Low"].isna().sum())
    result["missing_close_count"] = int(work_df["Close"].isna().sum())
    result["missing_volume_count"] = int(work_df["Volume"].isna().sum())

    if (
        result["missing_open_count"]
        + result["missing_high_count"]
        + result["missing_low_count"]
        + result["missing_close_count"]
        + result["missing_volume_count"]
    ) > 0:
        issues.append("missing_critical_values")

    result["nonpositive_price_count"] = int((work_df[PRICE_COLUMNS] <= 0).sum().sum())
    result["negative_volume_count"] = int((work_df["Volume"] < 0).sum())

    if result["nonpositive_price_count"] > 0:
        issues.append("nonpositive_prices")
    if result["negative_volume_count"] > 0:
        issues.append("negative_volume")

    observed_dates = pd.DatetimeIndex(work_df["Date"]).normalize().unique()
    min_date = observed_dates.min()
    max_date = observed_dates.max()
    expected = trading_calendar[(trading_calendar >= min_date) & (trading_calendar <= max_date)]
    missing_dates = expected.difference(observed_dates)
    result["missing_trading_days_count"] = int(len(missing_dates))

    if result["missing_trading_days_count"] > max_missing_trading_days_warning:
        issues.append("excess_missing_trading_days")

    has_critical = any(
        issue.startswith(
            (
                "missing_file",
                "read_error",
                "missing_columns",
                "no_valid_rows",
                "invalid_dates",
                "duplicate_dates",
                "non_monotonic_dates",
            )
        )
        for issue in issues
    )

    if has_critical:
        result["status"] = "error"
    elif len(issues) > 0:
        result["status"] = "warning"
    else:
        result["status"] = "ok"

    result["issues"] = ";".join(issues)
    return result


def write_markdown_summary(summary: dict, issues_breakdown: dict, summary_md: Path) -> None:
    summary_md.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# S&P 500 Per-File QA Summary",
        "",
        f"- Symbols in mapping: {summary['symbols_in_mapping']}",
        f"- Files checked: {summary['files_checked']}",
        f"- OK: {summary['status_ok']}",
        f"- Warning: {summary['status_warning']}",
        f"- Error: {summary['status_error']}",
        "",
        "## Issue Counts",
    ]

    if issues_breakdown:
        for issue, count in sorted(issues_breakdown.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {issue}: {count}")
    else:
        lines.append("- none")

    lines.append("")
    summary_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {args.mapping_csv}")

    trading_calendar = load_trading_calendar(args.calendar_csv)
    mapping_df = pd.read_csv(args.mapping_csv)

    required_mapping_cols = {"symbol", "file_path", "status"}
    if not required_mapping_cols.issubset(mapping_df.columns):
        missing = sorted(required_mapping_cols.difference(mapping_df.columns))
        raise ValueError(f"Mapping CSV is missing required columns: {missing}")

    checked_df = mapping_df[mapping_df["status"].astype(str).str.lower() == "ok"].copy()

    results = []
    for _, row in checked_df.iterrows():
        results.append(
            validate_one_file(
                symbol=str(row["symbol"]),
                file_path=str(row["file_path"]),
                trading_calendar=trading_calendar,
                max_missing_trading_days_warning=args.max_missing_trading_days_warning,
            )
        )

    report_df = pd.DataFrame(results)
    report_df = report_df.sort_values(by=["status", "symbol"], ascending=[True, True])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(args.output_csv, index=False)

    status_counts = report_df["status"].value_counts().to_dict()
    issues_counter: dict[str, int] = {}
    for issues in report_df["issues"].fillna(""):
        for issue in [x for x in str(issues).split(";") if x]:
            issues_counter[issue] = issues_counter.get(issue, 0) + 1

    summary = {
        "symbols_in_mapping": int(len(mapping_df)),
        "files_checked": int(len(report_df)),
        "status_ok": int(status_counts.get("ok", 0)),
        "status_warning": int(status_counts.get("warning", 0)),
        "status_error": int(status_counts.get("error", 0)),
        "calendar_min": trading_calendar.min().strftime("%Y-%m-%d"),
        "calendar_max": trading_calendar.max().strftime("%Y-%m-%d"),
        "output_csv": str(args.output_csv),
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps({"summary": summary, "issue_counts": issues_counter}, indent=2),
        encoding="utf-8",
    )

    write_markdown_summary(summary=summary, issues_breakdown=issues_counter, summary_md=args.summary_md)

    print("Validation complete.")
    print(f"Symbols in mapping: {summary['symbols_in_mapping']}")
    print(f"Files checked: {summary['files_checked']}")
    print(
        "Status counts: "
        f"ok={summary['status_ok']} warning={summary['status_warning']} error={summary['status_error']}"
    )
    print(f"QA CSV: {args.output_csv}")
    print(f"Summary JSON: {args.summary_json}")
    print(f"Summary Markdown: {args.summary_md}")


if __name__ == "__main__":
    main()
