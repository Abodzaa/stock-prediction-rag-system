# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- PatchTST | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009670, dir_acc=0.5185, sharpe=1.0402
- PatchTST | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009792, dir_acc=0.4947, sharpe=0.2232
- Informer | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010220, dir_acc=0.5000, sharpe=-0.4582
- PatchTST | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010352, dir_acc=0.4921, sharpe=-0.9674
- PatchTST | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010373, dir_acc=0.5503, sharpe=0.9070
- Informer | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010381, dir_acc=0.5132, sharpe=-0.8147
- Informer | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010576, dir_acc=0.4762, sharpe=-0.5898
- Informer | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010827, dir_acc=0.4656, sharpe=-0.2351
- PatchTST | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.019023, dir_acc=0.4894, sharpe=-0.4117
- Informer | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.036215, dir_acc=0.5079, sharpe=0.3027

## Baselines (Top 15 by RMSE)
- LSTM | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009618, dir_acc=0.5317, sharpe=1.0694
- LSTM | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009669, dir_acc=0.5212, sharpe=-0.1236
- LSTM | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009676, dir_acc=0.4709, sharpe=-1.0531
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009990, dir_acc=0.4974, sharpe=-0.1558
- LSTM | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010171, dir_acc=0.5106, sharpe=-0.0190
- GRU | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010215, dir_acc=0.4788, sharpe=-0.5844
- GRU | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010619, dir_acc=0.4630, sharpe=-1.0784
- GRU | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010627, dir_acc=0.4947, sharpe=-0.7822
- LSTM | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.017449, dir_acc=0.5106, sharpe=0.6764
- GRU | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.064217, dir_acc=0.4921, sharpe=-0.4624

## Failed / Skipped
- none
