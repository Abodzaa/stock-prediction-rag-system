# Panel Walk-Forward Experiment Summary

- Target: target_h1

## Feature Group Results
- G1_price_only: rmse=0.020387, mae=0.014010, dir_acc=0.5125, sharpe=0.1418, n_features=8
- G2_price_technical: rmse=0.020387, mae=0.014010, dir_acc=0.5125, sharpe=0.1418, n_features=17
- G3_plus_panel_breadth: rmse=0.020387, mae=0.014010, dir_acc=0.5125, sharpe=0.1400, n_features=22

## Model Family Sweep (Top 20 by RMSE)
- elastic_net | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.020387, mae=0.014010, dir_acc=0.5125, sharpe=0.1418
- elastic_net | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.020387, mae=0.014010, dir_acc=0.5125, sharpe=0.1418
- elastic_net | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.020387, mae=0.014010, dir_acc=0.5125, sharpe=0.1400
- hist_gbr | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.020387, mae=0.014009, dir_acc=0.5123, sharpe=0.1267
- ridge | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.020388, mae=0.014010, dir_acc=0.5110, sharpe=0.0991
- ridge | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.020388, mae=0.014011, dir_acc=0.5108, sharpe=0.1091
- hist_gbr | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.020389, mae=0.014011, dir_acc=0.5118, sharpe=0.1010
- ridge | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.020390, mae=0.014013, dir_acc=0.5088, sharpe=0.0900
- hist_gbr | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.020394, mae=0.014017, dir_acc=0.5097, sharpe=0.0943

## Model Set A-E
- Model_A_price_only: rmse=0.020387, mae=0.014010, dir_acc=0.5125, sharpe=0.1418, family=elastic_net, lag=0, epochs=120, patience=5, n_features=8
- Model_B_price_full_available: rmse=0.020387, mae=0.014010, dir_acc=0.5125, sharpe=0.1400, family=elastic_net, lag=0, epochs=120, patience=5, n_features=22
- Model_C_sentiment_only: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_D_price_shuffled_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_E_price_lagged_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
