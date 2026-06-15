# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- none

## Baselines (Top 15 by RMSE)
- LSTM | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.011627, dir_acc=0.5119, sharpe=-0.5097
- GRU | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.011721, dir_acc=0.4802, sharpe=-0.2080
- LSTM | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.012602, dir_acc=0.5119, sharpe=-0.5282
- GRU | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.016342, dir_acc=0.5079, sharpe=-0.6827

## Failed / Skipped
- none
