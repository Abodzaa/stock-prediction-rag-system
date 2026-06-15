# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- none

## Baselines (Top 15 by RMSE)
- GRU | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.010840, dir_acc=0.5000, sharpe=1.0065
- LSTM | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.012973, dir_acc=0.5119, sharpe=-0.4351
- LSTM | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.015363, dir_acc=0.5040, sharpe=0.2092
- GRU | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.039724, dir_acc=0.5000, sharpe=-0.8937

## Failed / Skipped
- none
