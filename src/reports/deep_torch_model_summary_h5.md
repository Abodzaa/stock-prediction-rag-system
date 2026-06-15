# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- FEDformer | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.015699, dir_acc=0.6349, sharpe=5.2681
- PatchTST | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.016168, dir_acc=0.6508, sharpe=2.6397
- Informer | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.016276, dir_acc=0.7302, sharpe=5.2635
- Informer | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.016459, dir_acc=0.6825, sharpe=3.9342
- PatchTST | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.016460, dir_acc=0.5873, sharpe=2.2694
- PatchTST | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.016490, dir_acc=0.6349, sharpe=2.3032
- TFT | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.016802, dir_acc=0.6508, sharpe=2.6397
- FEDformer | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.017112, dir_acc=0.5397, sharpe=-0.1119
- TFT | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.017605, dir_acc=0.6032, sharpe=1.5165
- MambaStock | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.017870, dir_acc=0.5714, sharpe=1.6311
- Informer | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.017951, dir_acc=0.3651, sharpe=-2.0813
- TFT | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.017978, dir_acc=0.4762, sharpe=-0.6204
- FEDformer | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.019180, dir_acc=0.6032, sharpe=-0.1589
- MambaStock | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.019202, dir_acc=0.5238, sharpe=-0.5081
- MambaStock | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.019271, dir_acc=0.5714, sharpe=0.1384

## Baselines (Top 15 by RMSE)
- GRU | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.016602, dir_acc=0.6190, sharpe=4.2228
- GRU | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.018617, dir_acc=0.4762, sharpe=-0.5412
- LSTM | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.018711, dir_acc=0.4762, sharpe=-2.6060
- NHITS | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.019042, dir_acc=0.5873, sharpe=4.0371
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.019099, dir_acc=0.4762, sharpe=-2.1999
- NHITS | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.020409, dir_acc=0.6190, sharpe=2.3180
- NHITS | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.022133, dir_acc=0.6032, sharpe=1.5628
- NBEATS | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.034849, dir_acc=0.5556, sharpe=1.5855
- LSTM | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.035769, dir_acc=0.3492, sharpe=-2.6397
- NBEATS | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.065825, dir_acc=0.4762, sharpe=-1.3753
- NBEATS | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.066830, dir_acc=0.4286, sharpe=-3.4336
- LSTM | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.083405, dir_acc=0.6508, sharpe=2.6397

## Failed / Skipped
- none
