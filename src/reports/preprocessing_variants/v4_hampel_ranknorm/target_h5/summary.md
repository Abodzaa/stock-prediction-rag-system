# Panel Walk-Forward Experiment Summary

- Target: target_h5

## Feature Group Results
- G1_price_only: rmse=0.046382, mae=0.032968, dir_acc=0.5220, sharpe=0.4187, n_features=8
- G3_plus_panel_breadth: rmse=0.046383, mae=0.032972, dir_acc=0.5193, sharpe=0.3852, n_features=22
- G2_price_technical: rmse=0.046384, mae=0.032972, dir_acc=0.5195, sharpe=0.3763, n_features=17

## Model Family Sweep (Top 20 by RMSE)
- ridge | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.046382, mae=0.032968, dir_acc=0.5220, sharpe=0.4187
- ridge | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.046383, mae=0.032972, dir_acc=0.5193, sharpe=0.3852
- ridge | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.046384, mae=0.032972, dir_acc=0.5195, sharpe=0.3763
- elastic_net | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.046390, mae=0.032973, dir_acc=0.5200, sharpe=0.3106
- elastic_net | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.046391, mae=0.032973, dir_acc=0.5206, sharpe=0.3426
- elastic_net | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.046391, mae=0.032973, dir_acc=0.5204, sharpe=0.3333
- hist_gbr | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.046392, mae=0.032975, dir_acc=0.5196, sharpe=0.3011
- hist_gbr | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.046402, mae=0.032977, dir_acc=0.5203, sharpe=0.3235
- hist_gbr | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.047261, mae=0.033638, dir_acc=0.4853, sharpe=-0.7218

## Model Set A-E
- Model_A_price_only: rmse=0.046382, mae=0.032968, dir_acc=0.5220, sharpe=0.4187, family=ridge, lag=0, epochs=120, patience=5, n_features=8
- Model_B_price_full_available: rmse=0.046383, mae=0.032972, dir_acc=0.5193, sharpe=0.3852, family=ridge, lag=0, epochs=120, patience=5, n_features=22
- Model_C_sentiment_only: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_D_price_shuffled_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_E_price_lagged_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
