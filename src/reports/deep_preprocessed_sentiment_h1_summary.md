# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- PatchTST | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009569, dir_acc=0.5423, sharpe=1.4768
- TFT | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009634, dir_acc=0.5000, sharpe=0.0708
- TFT | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009645, dir_acc=0.5000, sharpe=-0.2544
- TFT | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009771, dir_acc=0.5344, sharpe=-0.5087
- PatchTST | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009807, dir_acc=0.5106, sharpe=0.0013
- FEDformer | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009837, dir_acc=0.4947, sharpe=-0.0694
- PatchTST | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010002, dir_acc=0.5265, sharpe=0.9152
- Informer | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010043, dir_acc=0.5185, sharpe=0.1611
- Informer | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010246, dir_acc=0.5291, sharpe=1.0135
- Informer | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010386, dir_acc=0.4656, sharpe=-0.8697
- PatchTST | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010391, dir_acc=0.5000, sharpe=0.3722
- FEDformer | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010433, dir_acc=0.4709, sharpe=-0.2137
- Informer | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010633, dir_acc=0.4788, sharpe=-1.0619
- TFT | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010715, dir_acc=0.5265, sharpe=0.5395
- FEDformer | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011359, dir_acc=0.4735, sharpe=-1.3750

## Baselines (Top 15 by RMSE)
- LSTM | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009612, dir_acc=0.5370, sharpe=1.8883
- LSTM | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009750, dir_acc=0.5265, sharpe=-0.1417
- LSTM | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009837, dir_acc=0.4868, sharpe=-0.5918
- GRU | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009920, dir_acc=0.5185, sharpe=0.4097
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009959, dir_acc=0.5079, sharpe=-0.3731
- GRU | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010133, dir_acc=0.4630, sharpe=-0.8452
- LSTM | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010584, dir_acc=0.4762, sharpe=-0.2483
- GRU | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.015273, dir_acc=0.4841, sharpe=-0.1766
- LSTM | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.015803, dir_acc=0.5106, sharpe=0.6764
- GRU | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.037349, dir_acc=0.4947, sharpe=-0.0890

## Failed / Skipped
- none
