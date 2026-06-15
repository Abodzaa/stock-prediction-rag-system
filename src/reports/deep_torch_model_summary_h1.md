# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- Informer | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.007739, dir_acc=0.5714, sharpe=3.1570
- TFT | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.007805, dir_acc=0.5714, sharpe=0.5232
- TFT | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.007835, dir_acc=0.5873, sharpe=1.2702
- PatchTST | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.007876, dir_acc=0.4921, sharpe=2.0258
- PatchTST | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.007905, dir_acc=0.5714, sharpe=4.0843
- PatchTST | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.007908, dir_acc=0.5873, sharpe=0.6219
- TFT | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.008009, dir_acc=0.5873, sharpe=0.6876
- FEDformer | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.008216, dir_acc=0.5238, sharpe=1.1431
- Informer | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.008319, dir_acc=0.5714, sharpe=0.5346
- FEDformer | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.008666, dir_acc=0.4286, sharpe=-0.7758
- MambaStock | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.008930, dir_acc=0.4444, sharpe=-1.2952
- MambaStock | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009032, dir_acc=0.5873, sharpe=0.7824
- FEDformer | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009290, dir_acc=0.4127, sharpe=-0.6219
- MambaStock | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010934, dir_acc=0.6190, sharpe=3.9053
- Informer | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011554, dir_acc=0.4286, sharpe=-0.2911

## Baselines (Top 15 by RMSE)
- GRU | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.008007, dir_acc=0.6508, sharpe=3.8428
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.008111, dir_acc=0.5714, sharpe=0.7405
- NHITS | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.008393, dir_acc=0.5397, sharpe=0.9271
- GRU | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.008514, dir_acc=0.5556, sharpe=0.6673
- NHITS | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.008881, dir_acc=0.4286, sharpe=-1.4070
- NHITS | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011247, dir_acc=0.3968, sharpe=-2.7601
- LSTM | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011439, dir_acc=0.5873, sharpe=0.6219
- LSTM | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.020372, dir_acc=0.4127, sharpe=-0.6219
- NBEATS | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.036602, dir_acc=0.4444, sharpe=-1.5061
- NBEATS | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.057376, dir_acc=0.5397, sharpe=-0.8649
- NBEATS | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.067226, dir_acc=0.4286, sharpe=-1.1172
- LSTM | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.089360, dir_acc=0.5873, sharpe=0.6219

## Failed / Skipped
- none
