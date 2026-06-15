# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- PatchTST | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010767, dir_acc=0.5246, sharpe=0.9530
- Informer | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011174, dir_acc=0.4984, sharpe=0.1472
- Informer | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011423, dir_acc=0.4952, sharpe=0.2564
- Informer | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011459, dir_acc=0.5452, sharpe=0.9099
- PatchTST | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011527, dir_acc=0.5111, sharpe=0.3678
- Informer | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011547, dir_acc=0.5222, sharpe=0.0919
- PatchTST | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011770, dir_acc=0.4810, sharpe=0.0343
- PatchTST | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011920, dir_acc=0.4984, sharpe=-0.3616
- Informer | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.012294, dir_acc=0.5151, sharpe=0.0439
- PatchTST | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.016348, dir_acc=0.5151, sharpe=0.0682

## Baselines (Top 15 by RMSE)
- LSTM | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010816, dir_acc=0.5167, sharpe=-0.0131
- LSTM | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010911, dir_acc=0.5262, sharpe=0.2610
- LSTM | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010919, dir_acc=0.5167, sharpe=0.2016
- LSTM | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010941, dir_acc=0.5143, sharpe=-0.0170
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010959, dir_acc=0.5016, sharpe=-0.1220
- LSTM | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011001, dir_acc=0.5214, sharpe=0.0990
- GRU | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011683, dir_acc=0.4984, sharpe=-0.4840
- GRU | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011788, dir_acc=0.5175, sharpe=0.1057
- GRU | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011800, dir_acc=0.4825, sharpe=-0.2936
- GRU | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.012823, dir_acc=0.5119, sharpe=0.6484

## Failed / Skipped
- none
