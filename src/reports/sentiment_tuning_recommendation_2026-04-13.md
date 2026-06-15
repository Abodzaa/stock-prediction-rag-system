# Sentiment Tuning Recommendation (2026-04-13)

## Scope
- Pipeline policy: SBERT only for RAG challenger (Gemini removed from training due quota/permission limits).
- Targets tested: target_h1 and target_h5.
- Comparison rule: minimum Model C RMSE from sentiment pipeline comparison report.

## Completed Runs
1. Baseline SBERT full dataset runs (already completed earlier):
- reports/sentiment_pipeline_comparison_full_h1_sbert.csv
- reports/sentiment_pipeline_comparison_full_h5_sbert.csv

2. Targeted hyper-tuning runs completed in this session:
- reports/sentiment_pipeline_comparison_full_h1_sbert_tuned_targeted.csv
- reports/sentiment_pipeline_comparison_full_h5_sbert_tuned_targeted.csv
- Settings: models=elastic_net,ridge,hist_gbr, lag_window_grid=0, epochs_grid=150,300,600,900, patience_grid=10,20,40
- Runtime optimization for tuned pass: max_train_rows=200000, max_test_rows=80000

## Metric Comparison (Model C)
### target_h1
- Baseline SBERT: RMSE=0.0203874207880544, dir_acc=0.5125385771729373, sharpe=0.1417613000865053
- Tuned targeted: RMSE=0.0203878372048968, dir_acc=0.5125385771729373, sharpe=0.1417613000865053
- Result: tuned run is slightly worse on RMSE.

### target_h5
- Baseline SBERT: RMSE=0.0464271319258867, dir_acc=0.5200282589411396, sharpe=0.3105955759828637
- Tuned targeted: RMSE=0.0464287782500569, dir_acc=0.5200282589411396, sharpe=0.3105955759828637
- Result: tuned run is slightly worse on RMSE.

## Validation
- Current RAG article modes in data/processed/features/news_sentiment_rag_articles.csv:
  - sbert: 4242
- This confirms the active RAG sentiment features are SBERT-generated (no Gemini fallback in current training features).

## Recommendation
- Keep the original baseline SBERT full runs as production reference because they have the best RMSE in this test cycle:
  - reports/sentiment_pipeline_comparison_full_h1_sbert.md
  - reports/sentiment_pipeline_comparison_full_h5_sbert.md

- Use targeted tuning only as a periodic exploration workflow (reduced rows first, then full validation), because broad full-row sweeps are very expensive.
