# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- Informer | G1_price_only | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.016653, dir_acc=0.6508, sharpe=2.9515
- TFT | G2_price_technical | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.016732, dir_acc=0.6508, sharpe=2.6397
- TFT | G1_price_only | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.016944, dir_acc=0.6508, sharpe=2.6397
- TFT | G3_plus_breadth | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.017699, dir_acc=0.6508, sharpe=2.6397
- FEDformer | G3_plus_breadth | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.017819, dir_acc=0.4444, sharpe=-2.1254
- Informer | G3_plus_breadth | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.018195, dir_acc=0.6349, sharpe=1.8578
- PatchTST | G1_price_only | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.018583, dir_acc=0.5079, sharpe=-0.2726
- Informer | G2_price_technical | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.019347, dir_acc=0.4921, sharpe=0.3775
- FEDformer | G2_price_technical | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.020140, dir_acc=0.3651, sharpe=-3.1450
- FEDformer | G1_price_only | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.020584, dir_acc=0.5714, sharpe=0.2277
- Mamba | G2_price_technical | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.026282, dir_acc=0.5397, sharpe=0.8726
- PatchTST | G3_plus_breadth | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.027444, dir_acc=0.3333, sharpe=-3.6862
- Mamba | G1_price_only | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.028875, dir_acc=0.5556, sharpe=0.5320
- Mamba | G3_plus_breadth | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.028992, dir_acc=0.5714, sharpe=0.5450
- PatchTST | G2_price_technical | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.032368, dir_acc=0.4286, sharpe=-2.1211

## Baselines (Top 15 by RMSE)
- GRU | G1_price_only | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.016421, dir_acc=0.6508, sharpe=2.6397
- GRU | G2_price_technical | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.018482, dir_acc=0.6032, sharpe=2.8026
- GRU | G3_plus_breadth | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.018720, dir_acc=0.6032, sharpe=2.5709
- NHITS | G1_price_only | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.019952, dir_acc=0.4921, sharpe=-0.9708
- NHITS | G3_plus_breadth | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.021844, dir_acc=0.4286, sharpe=-1.3864
- NHITS | G2_price_technical | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.027063, dir_acc=0.4444, sharpe=-2.2191
- NBEATS | G1_price_only | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.039961, dir_acc=0.4286, sharpe=-1.7461
- NBEATS | G3_plus_breadth | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.053686, dir_acc=0.5714, sharpe=2.9915
- LSTM | G1_price_only | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.059259, dir_acc=0.6508, sharpe=2.6397
- NBEATS | G2_price_technical | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.059497, dir_acc=0.4762, sharpe=0.4565
- LSTM | G3_plus_breadth | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.110607, dir_acc=0.3492, sharpe=-2.6397
- LSTM | G2_price_technical | lag=32 | hidden=16 | lr=0.00100 | ep=5 | pat=3: rmse=0.175364, dir_acc=0.6508, sharpe=2.6397

## Failed / Skipped
- none
