# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- TFT | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.020391, dir_acc=0.5079, sharpe=-0.1666
- TFT | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.020946, dir_acc=0.4709, sharpe=-1.6376
- TFT | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.021164, dir_acc=0.4709, sharpe=-1.2586
- TFT | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.021185, dir_acc=0.4312, sharpe=-2.0017
- FEDformer | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.022496, dir_acc=0.4392, sharpe=-2.5034
- Informer | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.022770, dir_acc=0.4365, sharpe=-0.7389
- Informer | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.023377, dir_acc=0.4894, sharpe=-2.0455
- PatchTST | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.023483, dir_acc=0.5212, sharpe=0.5647
- PatchTST | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.023729, dir_acc=0.4392, sharpe=-1.9968
- Informer | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.023891, dir_acc=0.4762, sharpe=-1.1198
- PatchTST | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.024230, dir_acc=0.4233, sharpe=-2.7773
- Informer | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.024291, dir_acc=0.4603, sharpe=-1.7527
- FEDformer | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.025278, dir_acc=0.4444, sharpe=0.4348
- FEDformer | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.025816, dir_acc=0.4630, sharpe=-2.2877
- PatchTST | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.027561, dir_acc=0.4153, sharpe=-2.8053

## Baselines (Top 15 by RMSE)
- LSTM | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.020041, dir_acc=0.6058, sharpe=2.2119
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.020836, dir_acc=0.5397, sharpe=0.0853
- LSTM | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.021382, dir_acc=0.4683, sharpe=-0.4743
- LSTM | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.022255, dir_acc=0.4868, sharpe=-0.3417
- LSTM | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.022703, dir_acc=0.4312, sharpe=-2.1612
- LSTM | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.022792, dir_acc=0.4497, sharpe=-1.8095
- GRU | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.023324, dir_acc=0.4788, sharpe=-1.9016
- GRU | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.023482, dir_acc=0.4577, sharpe=-1.9326
- GRU | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.035115, dir_acc=0.4471, sharpe=-2.6157
- GRU | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.053531, dir_acc=0.4312, sharpe=-2.1612

## Failed / Skipped
- none
