# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- none

## Baselines (Top 15 by RMSE)
- LSTM | G1_price_only | lag=64 | hidden=64 | lr=0.00100 | ep=20 | pat=15: rmse=0.010874, dir_acc=0.5397, sharpe=0.6097
- LSTM | G1_price_only | lag=32 | hidden=32 | lr=0.00100 | ep=40 | pat=8: rmse=0.010924, dir_acc=0.5053, sharpe=0.3569
- LSTM | G1_price_only | lag=32 | hidden=32 | lr=0.00100 | ep=20 | pat=8: rmse=0.010924, dir_acc=0.5053, sharpe=0.3569
- LSTM | G1_price_only | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=15: rmse=0.010950, dir_acc=0.5265, sharpe=0.0571
- LSTM | G1_price_only | lag=64 | hidden=64 | lr=0.00050 | ep=40 | pat=8: rmse=0.011005, dir_acc=0.5238, sharpe=-0.2086
- LSTM | G1_price_only | lag=64 | hidden=64 | lr=0.00050 | ep=20 | pat=8: rmse=0.011005, dir_acc=0.5238, sharpe=-0.2086
- LSTM | G1_price_only | lag=32 | hidden=32 | lr=0.00100 | ep=40 | pat=15: rmse=0.011022, dir_acc=0.4974, sharpe=-1.1074
- LSTM | G1_price_only | lag=32 | hidden=32 | lr=0.00100 | ep=20 | pat=15: rmse=0.011022, dir_acc=0.4974, sharpe=-1.1074
- LSTM | G1_price_only | lag=64 | hidden=64 | lr=0.00050 | ep=40 | pat=15: rmse=0.011029, dir_acc=0.5370, sharpe=-0.1908
- LSTM | G1_price_only | lag=64 | hidden=64 | lr=0.00050 | ep=20 | pat=15: rmse=0.011029, dir_acc=0.5370, sharpe=-0.1908
- LSTM | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=40 | pat=15: rmse=0.011057, dir_acc=0.5212, sharpe=0.2256
- LSTM | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=15: rmse=0.011084, dir_acc=0.5212, sharpe=0.1629
- LSTM | G1_price_only | lag=32 | hidden=64 | lr=0.00100 | ep=20 | pat=15: rmse=0.011111, dir_acc=0.4868, sharpe=-1.5341
- LSTM | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=8: rmse=0.011113, dir_acc=0.5079, sharpe=0.3219
- LSTM | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=40 | pat=8: rmse=0.011113, dir_acc=0.5079, sharpe=0.3219

## Failed / Skipped
- none
