# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- PatchTST | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010938, dir_acc=0.5317, sharpe=0.9964
- PatchTST | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011029, dir_acc=0.5185, sharpe=0.7732
- Informer | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011136, dir_acc=0.5291, sharpe=-0.0942
- PatchTST | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011140, dir_acc=0.5397, sharpe=1.3418
- Informer | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011494, dir_acc=0.5159, sharpe=0.3259
- Informer | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011913, dir_acc=0.4788, sharpe=-0.2830
- PatchTST | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011987, dir_acc=0.4497, sharpe=-1.3690
- Informer | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.012003, dir_acc=0.4603, sharpe=-2.0345
- PatchTST | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.026562, dir_acc=0.4815, sharpe=-1.7889
- Informer | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.039408, dir_acc=0.4921, sharpe=-1.5252

## Baselines (Top 15 by RMSE)
- LSTM | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010940, dir_acc=0.5344, sharpe=0.6476
- LSTM | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010989, dir_acc=0.4788, sharpe=0.6328
- LSTM | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011061, dir_acc=0.5079, sharpe=-0.6032
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011219, dir_acc=0.5265, sharpe=-0.4065
- GRU | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011434, dir_acc=0.5317, sharpe=-0.2869
- GRU | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011462, dir_acc=0.5344, sharpe=0.2664
- GRU | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011736, dir_acc=0.4630, sharpe=-0.8362
- LSTM | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.013352, dir_acc=0.4709, sharpe=-1.7511
- LSTM | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.018362, dir_acc=0.4894, sharpe=-1.5702
- GRU | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.037706, dir_acc=0.4894, sharpe=-1.5702

## Failed / Skipped
- none
