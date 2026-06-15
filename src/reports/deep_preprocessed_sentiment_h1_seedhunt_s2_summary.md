# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- none

## Baselines (Top 15 by RMSE)
- LSTM | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.011733, dir_acc=0.4643, sharpe=-0.7421
- LSTM | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.013635, dir_acc=0.5000, sharpe=-0.8937
- GRU | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.016448, dir_acc=0.4960, sharpe=-0.4488
- GRU | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.034146, dir_acc=0.4960, sharpe=-1.1182

## Failed / Skipped
- none
