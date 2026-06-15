# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- TFT | G1_price_only | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.007943, dir_acc=0.5873, sharpe=1.7277
- TFT | G2_price_technical | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.007993, dir_acc=0.6508, sharpe=1.4403
- TFT | G3_plus_breadth | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.009154, dir_acc=0.5873, sharpe=0.6219
- Informer | G1_price_only | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.009417, dir_acc=0.6190, sharpe=2.2464
- FEDformer | G3_plus_breadth | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.010870, dir_acc=0.4921, sharpe=0.6546
- FEDformer | G1_price_only | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.014217, dir_acc=0.5238, sharpe=-2.0479
- PatchTST | G1_price_only | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.014235, dir_acc=0.4603, sharpe=2.1494
- Informer | G3_plus_breadth | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.016198, dir_acc=0.5556, sharpe=-0.8935
- FEDformer | G2_price_technical | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.017822, dir_acc=0.4444, sharpe=-1.1784
- PatchTST | G2_price_technical | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.018186, dir_acc=0.4286, sharpe=-1.2440
- PatchTST | G3_plus_breadth | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.018900, dir_acc=0.5397, sharpe=2.0476
- Mamba | G2_price_technical | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.025506, dir_acc=0.4444, sharpe=-0.9839
- Informer | G2_price_technical | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.028168, dir_acc=0.4603, sharpe=-1.7749
- Mamba | G1_price_only | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.029784, dir_acc=0.5238, sharpe=1.1919
- Mamba | G3_plus_breadth | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.035462, dir_acc=0.5714, sharpe=0.1664

## Baselines (Top 15 by RMSE)
- GRU | G1_price_only | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.008678, dir_acc=0.6190, sharpe=2.5795
- NHITS | G1_price_only | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.011162, dir_acc=0.4762, sharpe=0.5266
- GRU | G3_plus_breadth | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.012074, dir_acc=0.4603, sharpe=0.1801
- GRU | G2_price_technical | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.014692, dir_acc=0.5079, sharpe=0.3464
- NHITS | G3_plus_breadth | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.016418, dir_acc=0.4603, sharpe=-0.4425
- NHITS | G2_price_technical | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.021572, dir_acc=0.4286, sharpe=-1.7989
- NBEATS | G1_price_only | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.050905, dir_acc=0.4921, sharpe=-0.2678
- LSTM | G1_price_only | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.067799, dir_acc=0.5873, sharpe=0.6219
- NBEATS | G2_price_technical | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.070194, dir_acc=0.4603, sharpe=-0.3372
- NBEATS | G3_plus_breadth | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.081831, dir_acc=0.5873, sharpe=0.1843
- LSTM | G3_plus_breadth | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.125581, dir_acc=0.4127, sharpe=-0.6219
- LSTM | G2_price_technical | lag=32 | hidden=16 | lr=0.00100 | ep=3 | pat=2: rmse=0.186949, dir_acc=0.5873, sharpe=0.6219

## Failed / Skipped
- none
