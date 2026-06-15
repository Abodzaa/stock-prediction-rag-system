# Sentiment Pipeline Comparison

- Target: target_h1
- Selection rule: minimum Model C RMSE.

## Ranking
- finbert: C_rmse=0.0096592356552792, C_dir_acc=0.4833333333333333, C_sharpe=-0.5803911299975251, delta(C-D)_rmse=-0.0005470193644064002, delta(C-E)_rmse=-0.00035040041604390143
- rag: C_rmse=0.0096592356552792, C_dir_acc=0.4833333333333333, C_sharpe=-0.5803911299975251, delta(C-D)_rmse=-0.000509386418252801, delta(C-E)_rmse=-0.00035040041604390143

## Winner
- finbert

## Interpretation
- If C beats D (lower RMSE), sentiment carries non-random signal beyond shuffled control.
- If C beats E, contemporaneous sentiment is more informative than a lagged variant.
