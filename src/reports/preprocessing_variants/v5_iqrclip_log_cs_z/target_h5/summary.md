# Panel Walk-Forward Experiment Summary

- Target: target_h5

## Feature Group Results
- G2_price_technical: rmse=0.046418, mae=0.032993, dir_acc=0.5187, sharpe=0.2299, n_features=17
- G3_plus_panel_breadth: rmse=0.046422, mae=0.033009, dir_acc=0.5162, sharpe=0.2759, n_features=22
- G1_price_only: rmse=0.046426, mae=0.033000, dir_acc=0.5199, sharpe=0.3003, n_features=8

## Model Family Sweep (Top 20 by RMSE)
- hist_gbr | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.046418, mae=0.032993, dir_acc=0.5187, sharpe=0.2299
- ridge | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.046421, mae=0.033008, dir_acc=0.5162, sharpe=0.2577
- ridge | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.046422, mae=0.033009, dir_acc=0.5162, sharpe=0.2759
- elastic_net | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.046422, mae=0.032997, dir_acc=0.5200, sharpe=0.3030
- elastic_net | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.046423, mae=0.032998, dir_acc=0.5200, sharpe=0.2977
- elastic_net | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.046426, mae=0.033000, dir_acc=0.5199, sharpe=0.3003
- ridge | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.046429, mae=0.033005, dir_acc=0.5189, sharpe=0.2519
- hist_gbr | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.046430, mae=0.033001, dir_acc=0.5192, sharpe=0.2607
- hist_gbr | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.046443, mae=0.033009, dir_acc=0.5196, sharpe=0.2273

## Model Set A-E
- Model_A_price_only: rmse=0.046426, mae=0.033000, dir_acc=0.5199, sharpe=0.3003, family=elastic_net, lag=0, epochs=120, patience=5, n_features=8
- Model_B_price_full_available: rmse=0.046422, mae=0.033009, dir_acc=0.5162, sharpe=0.2759, family=ridge, lag=0, epochs=120, patience=5, n_features=22
- Model_C_sentiment_only: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_D_price_shuffled_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_E_price_lagged_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
