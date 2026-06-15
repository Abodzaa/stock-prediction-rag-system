# Sentiment Pipeline Comparison

- Target: target_h1
- Selection rule: minimum Model C RMSE.

## Ranking
- finbert: C_rmse=0.0203879790978675, C_dir_acc=0.5125385771729373, C_sharpe=0.1417613000865053, delta(C-D)_rmse=8.43856973269802e-06, delta(C-E)_rmse=6.137455976010231e-07
- rag: C_rmse=0.0203879790978675, C_dir_acc=0.5125385771729373, C_sharpe=0.1417613000865053, delta(C-D)_rmse=8.459548638598685e-06, delta(C-E)_rmse=6.137455976010231e-07

## Winner
- finbert

## Production Tracks
- baseline: finbert
- challenger: rag

## Interpretation
- If C beats D (lower RMSE), sentiment carries non-random signal beyond shuffled control.
- If C beats E, contemporaneous sentiment is more informative than a lagged variant.
