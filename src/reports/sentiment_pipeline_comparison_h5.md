# Sentiment Pipeline Comparison

- Target: target_h5
- Selection rule: minimum Model C RMSE.

## Ranking
- finbert: C_rmse=0.0175904899573726, C_dir_acc=0.3833333333333333, C_sharpe=-2.4752421315439017, delta(C-D)_rmse=-0.0008394635042592025, delta(C-E)_rmse=-0.0008916454652805016
- rag: C_rmse=0.0175904899573726, C_dir_acc=0.3833333333333333, C_sharpe=-2.4752421315439017, delta(C-D)_rmse=-0.0009035061335127027, delta(C-E)_rmse=-0.0008916454652805016

## Winner
- finbert

## Interpretation
- If C beats D (lower RMSE), sentiment carries non-random signal beyond shuffled control.
- If C beats E, contemporaneous sentiment is more informative than a lagged variant.
