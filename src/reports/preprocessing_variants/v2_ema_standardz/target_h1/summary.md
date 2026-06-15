# Panel Walk-Forward Experiment Summary

- Target: target_h1

## Feature Group Results
- G3_plus_panel_breadth: rmse=0.020322, mae=0.014001, dir_acc=0.5086, sharpe=0.2568, n_features=22
- G1_price_only: rmse=0.020376, mae=0.014002, dir_acc=0.5134, sharpe=0.2596, n_features=8
- G2_price_technical: rmse=0.020376, mae=0.014002, dir_acc=0.5134, sharpe=0.2596, n_features=17

## Model Family Sweep (Top 20 by RMSE)
- hist_gbr | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.020322, mae=0.014001, dir_acc=0.5086, sharpe=0.2568
- elastic_net | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.020376, mae=0.014002, dir_acc=0.5134, sharpe=0.2596
- elastic_net | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.020376, mae=0.014002, dir_acc=0.5134, sharpe=0.2596
- elastic_net | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.020376, mae=0.014002, dir_acc=0.5134, sharpe=0.2596
- ridge | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.020376, mae=0.014004, dir_acc=0.5124, sharpe=0.2650
- hist_gbr | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.020383, mae=0.014014, dir_acc=0.5129, sharpe=0.2810
- hist_gbr | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.020383, mae=0.014008, dir_acc=0.5143, sharpe=0.2662
- ridge | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.020389, mae=0.014032, dir_acc=0.5051, sharpe=0.2556
- ridge | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.020394, mae=0.014035, dir_acc=0.5045, sharpe=0.2446

## Model Set A-E
- Model_A_price_only: rmse=0.020376, mae=0.014002, dir_acc=0.5134, sharpe=0.2596, family=elastic_net, lag=0, epochs=120, patience=5, n_features=8
- Model_B_price_full_available: rmse=0.020322, mae=0.014001, dir_acc=0.5086, sharpe=0.2568, family=hist_gbr, lag=0, epochs=120, patience=5, n_features=22
- Model_C_sentiment_only: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_D_price_shuffled_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_E_price_lagged_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
