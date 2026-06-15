# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- PatchTST | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009677, dir_acc=0.5370, sharpe=1.5476
- Informer | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009937, dir_acc=0.4788, sharpe=-1.5655
- PatchTST | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009961, dir_acc=0.5132, sharpe=0.7927
- PatchTST | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010021, dir_acc=0.4894, sharpe=0.7122
- PatchTST | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010306, dir_acc=0.4974, sharpe=-0.6248
- Informer | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010469, dir_acc=0.4841, sharpe=-0.4088
- Informer | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010812, dir_acc=0.4815, sharpe=-1.0812
- Informer | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010909, dir_acc=0.4921, sharpe=-0.1622
- PatchTST | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.018224, dir_acc=0.4894, sharpe=-0.5972
- Informer | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.027246, dir_acc=0.4974, sharpe=-0.1725

## Baselines (Top 15 by RMSE)
- LSTM | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009565, dir_acc=0.5344, sharpe=0.4252
- LSTM | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009596, dir_acc=0.5159, sharpe=0.9565
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009714, dir_acc=0.5291, sharpe=0.6201
- LSTM | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009733, dir_acc=0.5026, sharpe=-0.1581
- GRU | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009836, dir_acc=0.4815, sharpe=-0.5214
- GRU | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009940, dir_acc=0.5265, sharpe=1.3556
- LSTM | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010992, dir_acc=0.4947, sharpe=0.1087
- GRU | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.013870, dir_acc=0.5106, sharpe=0.2827
- LSTM | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.015803, dir_acc=0.5106, sharpe=0.6764
- GRU | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.037349, dir_acc=0.4947, sharpe=-0.0890

## Failed / Skipped
- none
