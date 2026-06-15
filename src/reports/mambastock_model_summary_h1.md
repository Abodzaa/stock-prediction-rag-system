# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- MambaStock | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.008930, dir_acc=0.4444, sharpe=-1.2952
- MambaStock | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=5: rmse=0.008930, dir_acc=0.4444, sharpe=-1.2952
- MambaStock | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=5: rmse=0.009032, dir_acc=0.5873, sharpe=0.7824
- MambaStock | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009032, dir_acc=0.5873, sharpe=0.7824
- MambaStock | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00050 | ep=20 | pat=10: rmse=0.009220, dir_acc=0.4762, sharpe=-1.5750
- MambaStock | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00050 | ep=20 | pat=5: rmse=0.009220, dir_acc=0.4762, sharpe=-1.5750
- MambaStock | G1_price_only | lag=64 | hidden=32 | lr=0.00050 | ep=20 | pat=5: rmse=0.009333, dir_acc=0.5873, sharpe=0.6722
- MambaStock | G1_price_only | lag=64 | hidden=32 | lr=0.00050 | ep=20 | pat=10: rmse=0.009333, dir_acc=0.5873, sharpe=0.6722
- MambaStock | G3_plus_breadth | lag=32 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.009664, dir_acc=0.3968, sharpe=0.1126
- MambaStock | G2_price_technical | lag=64 | hidden=32 | lr=0.00050 | ep=20 | pat=10: rmse=0.010179, dir_acc=0.5556, sharpe=1.3441
- MambaStock | G1_price_only | lag=32 | hidden=32 | lr=0.00100 | ep=20 | pat=5: rmse=0.010356, dir_acc=0.4127, sharpe=-0.1473
- MambaStock | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010934, dir_acc=0.6190, sharpe=3.9053
- MambaStock | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=5: rmse=0.010934, dir_acc=0.6190, sharpe=3.9053
- MambaStock | G1_price_only | lag=32 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010938, dir_acc=0.5397, sharpe=0.4094
- MambaStock | G2_price_technical | lag=32 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011204, dir_acc=0.5556, sharpe=0.4698

## Baselines (Top 15 by RMSE)
- none

## Failed / Skipped
- none
