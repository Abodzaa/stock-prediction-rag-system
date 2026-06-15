# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- none

## Baselines (Top 15 by RMSE)
- GRU | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.013876, dir_acc=0.5119, sharpe=-0.7983
- LSTM | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.014095, dir_acc=0.5079, sharpe=-0.6241
- LSTM | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.032866, dir_acc=0.5000, sharpe=-0.8937
- GRU | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.052862, dir_acc=0.5000, sharpe=-0.8937

## Failed / Skipped
- none
