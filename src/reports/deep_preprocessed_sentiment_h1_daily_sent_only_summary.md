# Deep Torch Experiment Summary

## Primary Models (Top 15 by RMSE)
- PatchTST | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.010884, dir_acc=0.5212, sharpe=0.9581
- Informer | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011050, dir_acc=0.5079, sharpe=0.1948
- Informer | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011135, dir_acc=0.5212, sharpe=1.0355
- Informer | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011153, dir_acc=0.5212, sharpe=1.4311
- PatchTST | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011462, dir_acc=0.5238, sharpe=0.2838
- PatchTST | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011630, dir_acc=0.5106, sharpe=-1.2008
- Informer | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.012173, dir_acc=0.5556, sharpe=0.4814
- Informer | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.012486, dir_acc=0.5238, sharpe=-0.2119
- PatchTST | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.012609, dir_acc=0.4815, sharpe=-0.4316
- PatchTST | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.022534, dir_acc=0.5608, sharpe=0.6297

## Baselines (Top 15 by RMSE)
- GRU | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011313, dir_acc=0.5503, sharpe=0.9452
- LSTM | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011434, dir_acc=0.5132, sharpe=-0.0875
- LSTM | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011451, dir_acc=0.4815, sharpe=-0.0133
- LSTM | G2_price_technical | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011463, dir_acc=0.5079, sharpe=-1.0289
- GRU | G1_price_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011550, dir_acc=0.5132, sharpe=-1.0455
- LSTM | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011585, dir_acc=0.5185, sharpe=-0.5188
- GRU | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.011590, dir_acc=0.5000, sharpe=0.4865
- GRU | G3_plus_breadth | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.012140, dir_acc=0.4788, sharpe=0.1729
- LSTM | G4_price_plus_sentiment | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.012790, dir_acc=0.5291, sharpe=0.4084
- GRU | G5_sentiment_only | lag=64 | hidden=32 | lr=0.00100 | ep=20 | pat=10: rmse=0.014344, dir_acc=0.5714, sharpe=1.0245

## Failed / Skipped
- none
