# Sentiment Pipeline Comparison (H1 + H5, aligned data)

- Primary metric: Model C RMSE (sentiment-only).
- Secondary: directional accuracy, then control gaps vs Model D/E.

## Aggregate
- finbert: avg_c_rmse=0.013625, avg_c_dir_acc=0.4333, avg_c_sharpe=-1.5278, avg_delta_c_vs_d=-0.000693, avg_delta_c_vs_e=-0.000621
- rag: avg_c_rmse=0.013625, avg_c_dir_acc=0.4333, avg_c_sharpe=-1.5278, avg_delta_c_vs_d=-0.000706, avg_delta_c_vs_e=-0.000621

## Per Target
- h1 | finbert: c_rmse=0.009659, c_dir_acc=0.4833, c_sharpe=-0.5804, delta_c_vs_d=-0.000547, delta_c_vs_e=-0.000350
- h1 | rag: c_rmse=0.009659, c_dir_acc=0.4833, c_sharpe=-0.5804, delta_c_vs_d=-0.000509, delta_c_vs_e=-0.000350
- h5 | finbert: c_rmse=0.017590, c_dir_acc=0.3833, c_sharpe=-2.4752, delta_c_vs_d=-0.000839, delta_c_vs_e=-0.000892
- h5 | rag: c_rmse=0.017590, c_dir_acc=0.3833, c_sharpe=-2.4752, delta_c_vs_d=-0.000904, delta_c_vs_e=-0.000892

## Decision
- Winner by primary metric: tie
- Recommended production pipeline: finbert
- Reason: Model C performance is effectively tied; prefer FinBERT for simpler, domain-specific, and more stable production deployment.