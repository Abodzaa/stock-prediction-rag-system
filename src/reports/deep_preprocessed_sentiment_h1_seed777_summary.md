# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- PatchTST | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010843, dir_acc=0.5370, sharpe=1.2574
- PatchTST | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011104, dir_acc=0.4868, sharpe=-1.5292
- Informer | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011189, dir_acc=0.4788, sharpe=-0.5984
- PatchTST | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011261, dir_acc=0.5000, sharpe=0.3847
- Informer | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011275, dir_acc=0.5529, sharpe=0.6977
- Informer | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011338, dir_acc=0.4894, sharpe=-0.6828
- Informer | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011515, dir_acc=0.5079, sharpe=-0.9081
- Informer | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011924, dir_acc=0.5212, sharpe=0.3442
- PatchTST | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.022716, dir_acc=0.4841, sharpe=-0.2660
- PatchTST | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.047802, dir_acc=0.4921, sharpe=-1.4770

## Baselines (Top 15 by RMSE)
- LSTM | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010865, dir_acc=0.5132, sharpe=0.6654
- LSTM | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011057, dir_acc=0.5714, sharpe=1.0273
- LSTM | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011163, dir_acc=0.5423, sharpe=0.3231
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011207, dir_acc=0.5265, sharpe=-0.2079
- GRU | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011632, dir_acc=0.5185, sharpe=0.1135
- GRU | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.012887, dir_acc=0.4815, sharpe=-1.9339
- LSTM | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.012964, dir_acc=0.4868, sharpe=-1.7137
- LSTM | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.014342, dir_acc=0.5238, sharpe=0.0461
- GRU | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.015443, dir_acc=0.5661, sharpe=0.8467
- GRU | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.017929, dir_acc=0.4735, sharpe=-1.2214

## Failed / Skipped
- none
