# Sentiment Pipeline Comparison

- Target: target_h1
- Selection rule: minimum Model C RMSE.

## Ranking
- finbert: C_rmse=0.0203878372048968, C_dir_acc=0.5125385771729373, C_sharpe=0.1417613000865053, delta(C-D)_rmse=8.842163975699563e-06, delta(C-E)_rmse=-2.12905139799352e-07
- rag: C_rmse=0.0203878372048968, C_dir_acc=0.5125385771729373, C_sharpe=0.1417613000865053, delta(C-D)_rmse=8.845150087401221e-06, delta(C-E)_rmse=-2.12905139799352e-07

## Winner
- finbert

## Production Tracks
- baseline: finbert
- challenger: rag

## Interpretation
- If C beats D (lower RMSE), sentiment carries non-random signal beyond shuffled control.
- If C beats E, contemporaneous sentiment is more informative than a lagged variant.
