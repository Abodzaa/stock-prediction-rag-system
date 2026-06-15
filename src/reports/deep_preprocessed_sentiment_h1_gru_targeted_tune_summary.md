# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- none

## Baselines (Top 15 by RMSE)
- GRU | G1_price_only | lag=64 | hidden=64 | lr=0.00100 | ep=20 | pat=15: rmse=0.010946, dir_acc=0.5159, sharpe=0.5860
- GRU | G1_price_only | lag=32 | hidden=64 | lr=0.00100 | ep=40 | pat=15: rmse=0.010987, dir_acc=0.4577, sharpe=-0.2770
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00050 | ep=20 | pat=15: rmse=0.011011, dir_acc=0.4947, sharpe=-0.6304
- GRU | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=40 | pat=15: rmse=0.011038, dir_acc=0.5185, sharpe=0.6312
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=15: rmse=0.011076, dir_acc=0.5132, sharpe=-0.3642
- GRU | G1_price_only | lag=32 | hidden=64 | lr=0.00100 | ep=20 | pat=15: rmse=0.011079, dir_acc=0.4524, sharpe=-0.7975
- GRU | G1_price_only | lag=64 | hidden=64 | lr=0.00100 | ep=40 | pat=8: rmse=0.011084, dir_acc=0.5476, sharpe=0.6480
- GRU | G1_price_only | lag=64 | hidden=64 | lr=0.00100 | ep=20 | pat=8: rmse=0.011084, dir_acc=0.5476, sharpe=0.6480
- GRU | G1_price_only | lag=32 | hidden=64 | lr=0.00100 | ep=40 | pat=8: rmse=0.011119, dir_acc=0.4815, sharpe=-0.7319
- GRU | G1_price_only | lag=32 | hidden=64 | lr=0.00100 | ep=20 | pat=8: rmse=0.011119, dir_acc=0.4815, sharpe=-0.7319
- GRU | G1_price_only | lag=32 | hidden=32 | lr=0.00100 | ep=40 | pat=15: rmse=0.011134, dir_acc=0.5053, sharpe=0.2187
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00050 | ep=20 | pat=8: rmse=0.011141, dir_acc=0.5344, sharpe=0.6140
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00050 | ep=40 | pat=8: rmse=0.011141, dir_acc=0.5344, sharpe=0.6140
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=15: rmse=0.011155, dir_acc=0.5053, sharpe=-0.7322
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=40 | pat=15: rmse=0.011155, dir_acc=0.5053, sharpe=-0.7322

## Failed / Skipped
- none
