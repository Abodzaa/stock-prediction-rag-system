# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- PatchTST | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010890, dir_acc=0.5556, sharpe=0.7137
- PatchTST | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011061, dir_acc=0.5344, sharpe=1.5763
- Informer | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011088, dir_acc=0.4947, sharpe=-0.4205
- Informer | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011225, dir_acc=0.4921, sharpe=-0.6437
- PatchTST | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011321, dir_acc=0.5476, sharpe=0.6057
- Informer | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011337, dir_acc=0.5053, sharpe=-0.7474
- PatchTST | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011470, dir_acc=0.5159, sharpe=-0.6318
- Informer | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011820, dir_acc=0.4868, sharpe=-0.0216
- Informer | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.012173, dir_acc=0.5556, sharpe=0.4814
- PatchTST | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.022534, dir_acc=0.5608, sharpe=0.6297

## Baselines (Top 15 by RMSE)
- LSTM | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010926, dir_acc=0.5079, sharpe=0.1501
- GRU | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010952, dir_acc=0.5423, sharpe=0.9691
- LSTM | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011084, dir_acc=0.5212, sharpe=0.1629
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011155, dir_acc=0.5053, sharpe=-0.7322
- GRU | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011219, dir_acc=0.5000, sharpe=0.0641
- LSTM | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011451, dir_acc=0.4815, sharpe=-0.0133
- LSTM | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011494, dir_acc=0.5556, sharpe=0.4104
- GRU | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011951, dir_acc=0.5185, sharpe=0.6440
- LSTM | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.012870, dir_acc=0.4735, sharpe=-1.2159
- GRU | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.014344, dir_acc=0.5714, sharpe=1.0245

## Failed / Skipped
- none
