# Panel Walk-Forward Experiment Summary

- Target: target_h5

## Feature Group Results
- G3_plus_panel_breadth: rmse=0.046404, mae=0.033005, dir_acc=0.5137, sharpe=0.3259, n_features=22
- G1_price_only: rmse=0.046407, mae=0.032976, dir_acc=0.5171, sharpe=0.2504, n_features=8
- G2_price_technical: rmse=0.046418, mae=0.033021, dir_acc=0.5108, sharpe=0.3008, n_features=17

## Model Family Sweep (Top 20 by RMSE)
- elastic_net | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.046404, mae=0.033005, dir_acc=0.5137, sharpe=0.3259
- elastic_net | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.046407, mae=0.032976, dir_acc=0.5171, sharpe=0.2504
- ridge | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.046418, mae=0.032985, dir_acc=0.5137, sharpe=0.1673
- elastic_net | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.046418, mae=0.033021, dir_acc=0.5108, sharpe=0.3008
- hist_gbr | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.046487, mae=0.033029, dir_acc=0.5172, sharpe=0.2345
- ridge | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.046487, mae=0.033116, dir_acc=0.5100, sharpe=0.3040
- ridge | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.046494, mae=0.033127, dir_acc=0.5073, sharpe=0.2563
- hist_gbr | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.046589, mae=0.033134, dir_acc=0.5146, sharpe=0.3196
- hist_gbr | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.046746, mae=0.033266, dir_acc=0.5079, sharpe=0.1216

## Model Set A-E
- Model_A_price_only: rmse=0.046407, mae=0.032976, dir_acc=0.5171, sharpe=0.2504, family=elastic_net, lag=0, epochs=120, patience=5, n_features=8
- Model_B_price_full_available: rmse=0.046404, mae=0.033005, dir_acc=0.5137, sharpe=0.3259, family=elastic_net, lag=0, epochs=120, patience=5, n_features=22
- Model_C_sentiment_only: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_D_price_shuffled_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_E_price_lagged_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
