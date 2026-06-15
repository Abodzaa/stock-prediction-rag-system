# Sentiment Pipeline Comparison

- Target: target_h5
- Selection rule: minimum Model C RMSE.

## Ranking
- finbert: C_rmse=0.0464271319258867, C_dir_acc=0.5200282589411396, C_sharpe=0.3105955759828637, delta(C-D)_rmse=-6.416780471203698e-06, delta(C-E)_rmse=2.726377359395238e-06
- rag: C_rmse=0.0464271319258867, C_dir_acc=0.5200282589411396, C_sharpe=0.3105955759828637, delta(C-D)_rmse=-6.179008732905278e-06, delta(C-E)_rmse=2.726377359395238e-06

## Winner
- finbert

## Production Tracks
- baseline: finbert
- challenger: rag

## Interpretation
- If C beats D (lower RMSE), sentiment carries non-random signal beyond shuffled control.
- If C beats E, contemporaneous sentiment is more informative than a lagged variant.
