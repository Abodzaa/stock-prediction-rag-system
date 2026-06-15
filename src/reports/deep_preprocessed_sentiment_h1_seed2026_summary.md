# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- PatchTST | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010769, dir_acc=0.5238, sharpe=1.2312
- PatchTST | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010928, dir_acc=0.5000, sharpe=0.7833
- Informer | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011264, dir_acc=0.5291, sharpe=0.4751
- PatchTST | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011416, dir_acc=0.5159, sharpe=-0.7125
- Informer | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011692, dir_acc=0.4868, sharpe=-0.3001
- Informer | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011892, dir_acc=0.5079, sharpe=-1.2775
- PatchTST | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.013662, dir_acc=0.5000, sharpe=-1.5519
- Informer | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.014509, dir_acc=0.5661, sharpe=0.7864
- Informer | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.015531, dir_acc=0.4868, sharpe=-1.1499
- PatchTST | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.099539, dir_acc=0.4894, sharpe=-1.5702

## Baselines (Top 15 by RMSE)
- LSTM | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010997, dir_acc=0.5132, sharpe=0.4506
- LSTM | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011094, dir_acc=0.4868, sharpe=-0.7250
- LSTM | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011106, dir_acc=0.5238, sharpe=-0.4669
- GRU | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011133, dir_acc=0.5132, sharpe=-0.1404
- LSTM | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011204, dir_acc=0.5291, sharpe=0.2868
- GRU | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011313, dir_acc=0.5291, sharpe=-0.0553
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011743, dir_acc=0.5026, sharpe=-0.5600
- GRU | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.012151, dir_acc=0.5661, sharpe=1.0816
- LSTM | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.013121, dir_acc=0.4815, sharpe=-1.6991
- GRU | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.017208, dir_acc=0.4974, sharpe=-1.2950

## Failed / Skipped
- none
