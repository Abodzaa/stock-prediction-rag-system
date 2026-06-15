# Sentiment Pipeline Comparison

- Target: target_h5
- Selection rule: minimum Model C RMSE.

## Ranking
- finbert: C_rmse=0.0464257281328487, C_dir_acc=0.5200282589411396, C_sharpe=0.3105955759828637, delta(C-D)_rmse=-1.690025872504164e-06, delta(C-E)_rmse=1.303882795899991e-06
- rag: C_rmse=0.0464257281328487, C_dir_acc=0.5200282589411396, C_sharpe=0.3105955759828637, delta(C-D)_rmse=-1.4204642069040596e-06, delta(C-E)_rmse=1.303882795899991e-06

## Winner
- finbert

## Production Tracks
- baseline: finbert
- challenger: rag

## Interpretation
- If C beats D (lower RMSE), sentiment carries non-random signal beyond shuffled control.
- If C beats E, contemporaneous sentiment is more informative than a lagged variant.
