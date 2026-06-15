# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- PatchTST | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010765, dir_acc=0.5317, sharpe=1.3340
- PatchTST | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010914, dir_acc=0.5159, sharpe=-0.3094
- PatchTST | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011180, dir_acc=0.5529, sharpe=0.6229
- PatchTST | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011187, dir_acc=0.5265, sharpe=0.2642
- Informer | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011208, dir_acc=0.4788, sharpe=-0.7319
- Informer | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011390, dir_acc=0.4921, sharpe=-0.0292
- Informer | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011551, dir_acc=0.5238, sharpe=0.4888
- Informer | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.014642, dir_acc=0.4894, sharpe=-1.5702
- Informer | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.018552, dir_acc=0.4868, sharpe=-1.6432
- PatchTST | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.023240, dir_acc=0.4921, sharpe=-1.3351

## Baselines (Top 15 by RMSE)
- LSTM | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011015, dir_acc=0.5503, sharpe=0.4948
- LSTM | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011136, dir_acc=0.5132, sharpe=-0.3809
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011141, dir_acc=0.5423, sharpe=-0.0191
- GRU | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011233, dir_acc=0.4894, sharpe=0.0456
- LSTM | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011384, dir_acc=0.5397, sharpe=0.2405
- GRU | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011491, dir_acc=0.5397, sharpe=0.1145
- GRU | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011501, dir_acc=0.5608, sharpe=1.4651
- LSTM | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011753, dir_acc=0.4947, sharpe=-1.4016
- LSTM | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.012339, dir_acc=0.5344, sharpe=1.0279
- GRU | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.027551, dir_acc=0.4894, sharpe=-1.5702

## Failed / Skipped
- none
