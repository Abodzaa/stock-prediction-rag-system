# Stock Market Prediction

This repository contains data pipelines and experiment runners for S&P 500 forecasting with:

- price/technical features,
- cross-sectional breadth signals,
- optional news sentiment features (FinBERT or RAG-style),
- walk-forward model evaluation.

Processed datasets are intentionally ignored by Git. New users should recreate them locally with the commands below.

## Project Map

Top-level folders:

- `src/data/`: data ingestion and QA scripts.
- `src/features/`: feature engineering and preprocessing variants.
- `src/nlp/`: news collection, sentiment modeling, and sentiment-feature merging.
- `src/experiments/`: walk-forward experiments and model sweeps.
- `configs/`: research contract and project-level assumptions.
- `data/raw_market/`: raw index-level market data.
- `data/raw_equities/`: one file per constituent ticker.
- `data/raw_news/`: collected and merged news corpora.
- `data/processed/`: locally generated feature/sentiment datasets (gitignored).
- `data/metadata/`: run metadata, mappings, QA outputs.
- `reports/`: experiment summaries and CSV/JSON/MD outputs.
- `artifacts/models/`: trained model artifacts.

Key scripts to know:

- `src/data/download_sp500.py`: download index OHLCV (`^GSPC`).
- `src/data/download_sp500_constituents.py`: download per-constituent OHLCV files.
- `src/data/validate_sp500_constituents.py`: QA checks against market calendar.
- `src/features/build_research_features.py`: build daily research feature matrix.
- `src/features/build_equity_panel_features.py`: build panel (`Date`, `Symbol`) feature matrix.
- `src/features/run_preprocessing_variants.py`: generate v1-v5 preprocessing datasets.
- `src/nlp/fetch_yfinance_news.py`: fetch Yahoo Finance news.
- `src/nlp/fetch_gdelt_news.py`: fetch historical GDELT news.
- `src/nlp/merge_news_sources.py`: deduplicate and merge raw news files.
- `src/nlp/build_sentiment_finbert.py`: article and daily FinBERT sentiment features.
- `src/nlp/build_sentiment_rag.py`: article and daily RAG-style sentiment features.
- `src/nlp/merge_sentiment_features.py`: join daily sentiment into base daily features.

## Environment Setup

1. Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional deep-learning extras:

```powershell
pip install -r requirements-deep.txt
```

## Recreate Data (End-to-End)

Run from repository root.

### 1) Download index market data

```powershell
python src/data/download_sp500.py
```

Outputs:

- `data/raw_market/sp500_2006_2026.csv`
- `data/metadata/sp500_2006_2026_metadata.json`
- `data/metadata/sp500_2006_2026_sha256.txt`

### 2) Download S&P 500 constituent files

```powershell
python src/data/download_sp500_constituents.py
```

Outputs:

- `data/raw_equities/sp500_constituents/*.csv`
- `data/metadata/sp500_constituent_file_map.csv`
- `data/metadata/sp500_constituents_snapshot.csv`

### 3) Run constituent QA validation

```powershell
python src/data/validate_sp500_constituents.py
```

Outputs:

- `data/metadata/sp500_constituent_qa_report.csv`
- `data/metadata/sp500_constituent_qa_summary.json`
- `reports/sp500_constituent_qa_summary.md`

### 4) Build core feature datasets

Daily research features (index + breadth):

```powershell
python src/features/build_research_features.py
```

Panel features (`Date`, `Symbol`):

```powershell
python src/features/build_equity_panel_features.py
```

Outputs:

- `data/processed/features/research_features_daily.csv`
- `data/processed/features/research_features_panel.csv`
- corresponding metadata JSON files in `data/metadata/`

### 5) (Optional) Build preprocessing variants v1-v5

```powershell
python src/features/run_preprocessing_variants.py
```

This materializes the variant CSVs under `data/processed/features/`.

### 6) Collect and merge news corpus

Yahoo Finance news:

```powershell
python src/nlp/fetch_yfinance_news.py
```

GDELT news:

```powershell
python src/nlp/fetch_gdelt_news.py
```

Merge sources:

```powershell
python src/nlp/merge_news_sources.py --inputs data/raw_news/yfinance_sp500_news.csv,data/raw_news/gdelt_market_news_2015_2026.csv
```

Output:

- `data/raw_news/market_news_combined.csv`

### 7) Build sentiment features

FinBERT pipeline:

```powershell
python src/nlp/build_sentiment_finbert.py --input-csv data/raw_news/market_news_combined.csv
```

RAG-style pipeline (SBERT backend):

```powershell
python src/nlp/build_sentiment_rag.py --input-csv data/raw_news/market_news_combined.csv --backend sbert
```

Outputs are written to `data/processed/features/`, including article-level and daily sentiment files.

### 8) Merge daily sentiment into daily research features

Example using FinBERT daily features:

```powershell
python src/nlp/merge_sentiment_features.py --base-features data/processed/features/research_features_daily.csv --sentiment-daily data/processed/features/news_sentiment_finbert_daily.csv --output-features data/processed/features/research_features_with_sentiment.csv
```

## Minimal Repro Path (Fast)

If you only need a reproducible baseline dataset:

1. `python src/data/download_sp500.py`
2. `python src/data/download_sp500_constituents.py`
3. `python src/data/validate_sp500_constituents.py`
4. `python src/features/build_research_features.py`
5. (optional) `python src/features/build_equity_panel_features.py`

## Running Experiments

Examples:

Daily walk-forward baseline:

```powershell
python src/experiments/run_walkforward_experiments.py --features-csv data/processed/features/research_features_daily.csv --target target_h1
```

Panel walk-forward experiments:

```powershell
python src/experiments/run_panel_walkforward_experiments.py --features-csv data/processed/features/research_features_panel.csv --target target_h1
```

Sentiment pipeline comparison:

```powershell
python src/experiments/run_phase5_sentiment_comparison.py
```

## Data and Git Policy

Large generated datasets are excluded via `.gitignore` (for example `data/processed/`, `artifacts/`, and raw bulk data folders).

Expected workflow:

- clone repository,
- install dependencies,
- regenerate data locally using this README,
- run experiments.

## Troubleshooting

- If a download script fails intermittently, rerun it; scripts already include retries/backoff.
- If sentiment scripts are slow, reduce rows with `--max-rows` for smoke tests.
- If using Gemini backend in `build_sentiment_rag.py`, provide API key via `--gemini-api-key` or environment and expect rate limits on free tier.
