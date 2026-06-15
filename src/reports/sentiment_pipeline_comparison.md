# Sentiment Pipeline Comparison

- Target: target_h5
- Selection rule: minimum Model C RMSE.

## Ranking
- finbert: C_rmse=0.0170163897704569, C_dir_acc=0.6333333333333333, C_sharpe=2.12616262894767, delta(C-D)_rmse=-4.913114317489989e-05, delta(C-E)_rmse=-4.913114317489989e-05
- rag: C_rmse=0.0170163897704569, C_dir_acc=0.6333333333333333, C_sharpe=2.12616262894767, delta(C-D)_rmse=-4.913114317489989e-05, delta(C-E)_rmse=-4.913114317489989e-05

## Winner
- finbert

## Interpretation
- If C beats D (lower RMSE), sentiment carries non-random signal beyond shuffled control.
- If C beats E, contemporaneous sentiment is more informative than a lagged variant.
