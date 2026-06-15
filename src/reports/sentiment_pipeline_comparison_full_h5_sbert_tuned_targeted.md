# Sentiment Pipeline Comparison

- Target: target_h5
- Selection rule: minimum Model C RMSE.

## Ranking
- finbert: C_rmse=0.0464287782500569, C_dir_acc=0.5200282589411396, C_sharpe=0.3105955759828637, delta(C-D)_rmse=-5.72725517559991e-06, delta(C-E)_rmse=1.0518069937001662e-05
- rag: C_rmse=0.0464287782500569, C_dir_acc=0.5200282589411396, C_sharpe=0.3105955759828637, delta(C-D)_rmse=-5.415780186500407e-06, delta(C-E)_rmse=1.0518069937001662e-05

## Winner
- finbert

## Production Tracks
- baseline: finbert
- challenger: rag

## Interpretation
- If C beats D (lower RMSE), sentiment carries non-random signal beyond shuffled control.
- If C beats E, contemporaneous sentiment is more informative than a lagged variant.
