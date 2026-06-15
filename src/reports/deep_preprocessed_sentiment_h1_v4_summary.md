# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- PatchTST | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009537, dir_acc=0.5238, sharpe=-0.1980
- PatchTST | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009547, dir_acc=0.5079, sharpe=0.3027
- Informer | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009566, dir_acc=0.5079, sharpe=0.3027
- PatchTST | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009573, dir_acc=0.4894, sharpe=-0.3985
- Informer | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009576, dir_acc=0.5238, sharpe=-0.1980
- Informer | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009662, dir_acc=0.5079, sharpe=0.3027
- PatchTST | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.013790, dir_acc=0.4894, sharpe=-0.2456
- Informer | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.016285, dir_acc=0.4921, sharpe=-0.2227
- PatchTST | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.018224, dir_acc=0.4894, sharpe=-0.5972
- Informer | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.027246, dir_acc=0.4974, sharpe=-0.1725

## Baselines (Top 15 by RMSE)
- LSTM | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009531, dir_acc=0.5079, sharpe=0.3027
- LSTM | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009546, dir_acc=0.4921, sharpe=-0.4624
- GRU | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009546, dir_acc=0.5079, sharpe=0.3027
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009548, dir_acc=0.4921, sharpe=-0.4624
- LSTM | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009556, dir_acc=0.4921, sharpe=-0.4624
- GRU | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009571, dir_acc=0.4947, sharpe=-0.4380
- LSTM | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.014293, dir_acc=0.5238, sharpe=0.1695
- LSTM | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.015803, dir_acc=0.5106, sharpe=0.6764
- GRU | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.027825, dir_acc=0.5238, sharpe=0.7084
- GRU | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.037349, dir_acc=0.4947, sharpe=-0.0890

## Failed / Skipped
- none
