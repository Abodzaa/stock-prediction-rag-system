# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- none

## Baselines (Top 15 by RMSE)
- GRU | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.012506, dir_acc=0.5595, sharpe=1.0029
- LSTM | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.014707, dir_acc=0.4643, sharpe=-0.9678
- GRU | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.030222, dir_acc=0.5040, sharpe=-0.9220
- LSTM | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.039185, dir_acc=0.5000, sharpe=-0.8937

## Failed / Skipped
- none
