# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- PatchTST | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009552, dir_acc=0.4921, sharpe=-0.4624
- PatchTST | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009557, dir_acc=0.5000, sharpe=0.3135
- PatchTST | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009564, dir_acc=0.4921, sharpe=-0.4624
- Informer | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009576, dir_acc=0.4921, sharpe=-0.4624
- Informer | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009580, dir_acc=0.5000, sharpe=1.0863
- Informer | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009591, dir_acc=0.4921, sharpe=-0.4624
- Informer | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009854, dir_acc=0.4974, sharpe=-0.3701
- PatchTST | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.012230, dir_acc=0.5212, sharpe=0.6574
- PatchTST | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.018224, dir_acc=0.4894, sharpe=-0.5972
- Informer | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.027246, dir_acc=0.4974, sharpe=-0.1725

## Baselines (Top 15 by RMSE)
- GRU | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009527, dir_acc=0.5106, sharpe=0.4174
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009542, dir_acc=0.5079, sharpe=0.3027
- LSTM | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009542, dir_acc=0.5079, sharpe=0.3027
- LSTM | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009546, dir_acc=0.4921, sharpe=-0.4624
- GRU | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009547, dir_acc=0.4921, sharpe=-0.4624
- LSTM | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009569, dir_acc=0.4921, sharpe=-0.4513
- LSTM | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.012006, dir_acc=0.5000, sharpe=-0.1324
- LSTM | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.015803, dir_acc=0.5106, sharpe=0.6764
- GRU | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.037349, dir_acc=0.4947, sharpe=-0.0890
- GRU | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.050534, dir_acc=0.4868, sharpe=-0.2923

## Failed / Skipped
- none
