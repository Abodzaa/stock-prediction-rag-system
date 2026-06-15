# Panel Walk-Forward Experiment Summary

- Target: target_h5

## Feature Group Results
- G1_price_only: rmse=0.046411, mae=0.032978, dir_acc=0.5170, sharpe=0.2516, n_features=8
- G3_plus_panel_breadth: rmse=0.046432, mae=0.033006, dir_acc=0.5137, sharpe=0.1611, n_features=22
- G2_price_technical: rmse=0.046435, mae=0.033013, dir_acc=0.5118, sharpe=0.1487, n_features=17

## Model Family Sweep (Top 20 by RMSE)
- elastic_net | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.046411, mae=0.032978, dir_acc=0.5170, sharpe=0.2516
- elastic_net | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.046432, mae=0.033006, dir_acc=0.5137, sharpe=0.1611
- ridge | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.046433, mae=0.032997, dir_acc=0.5130, sharpe=0.1709
- elastic_net | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.046435, mae=0.033013, dir_acc=0.5118, sharpe=0.1487
- hist_gbr | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.046481, mae=0.033026, dir_acc=0.5170, sharpe=0.2268
- ridge | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.046508, mae=0.033071, dir_acc=0.5117, sharpe=0.1531
- ridge | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.046511, mae=0.033080, dir_acc=0.5116, sharpe=0.1790
- hist_gbr | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.046608, mae=0.033121, dir_acc=0.5098, sharpe=0.1021
- hist_gbr | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.046670, mae=0.033168, dir_acc=0.5088, sharpe=-0.0147

## Model Set A-E
- Model_A_price_only: rmse=0.046411, mae=0.032978, dir_acc=0.5170, sharpe=0.2516, family=elastic_net, lag=0, epochs=120, patience=5, n_features=8
- Model_B_price_full_available: rmse=0.046432, mae=0.033006, dir_acc=0.5137, sharpe=0.1611, family=elastic_net, lag=0, epochs=120, patience=5, n_features=22
- Model_C_sentiment_only: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_D_price_shuffled_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_E_price_lagged_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
