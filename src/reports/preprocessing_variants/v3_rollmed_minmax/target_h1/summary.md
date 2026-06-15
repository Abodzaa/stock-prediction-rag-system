# Panel Walk-Forward Experiment Summary

- Target: target_h1

## Feature Group Results
- G3_plus_panel_breadth: rmse=0.020379, mae=0.014013, dir_acc=0.5054, sharpe=0.2555, n_features=22
- G2_price_technical: rmse=0.020382, mae=0.014017, dir_acc=0.5056, sharpe=0.2673, n_features=17
- G1_price_only: rmse=0.020383, mae=0.014006, dir_acc=0.5092, sharpe=0.1605, n_features=8

## Model Family Sweep (Top 20 by RMSE)
- ridge | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.020379, mae=0.014013, dir_acc=0.5054, sharpe=0.2555
- ridge | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.020382, mae=0.014017, dir_acc=0.5056, sharpe=0.2673
- ridge | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.020383, mae=0.014006, dir_acc=0.5092, sharpe=0.1605
- elastic_net | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.020383, mae=0.014005, dir_acc=0.5158, sharpe=0.2159
- elastic_net | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.020385, mae=0.014008, dir_acc=0.5129, sharpe=0.1554
- elastic_net | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.020385, mae=0.014008, dir_acc=0.5129, sharpe=0.1554
- hist_gbr | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.020389, mae=0.014010, dir_acc=0.5114, sharpe=0.1246
- hist_gbr | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.020391, mae=0.014011, dir_acc=0.5107, sharpe=0.1063
- hist_gbr | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.020392, mae=0.014007, dir_acc=0.5113, sharpe=0.2154

## Model Set A-E
- Model_A_price_only: rmse=0.020383, mae=0.014006, dir_acc=0.5092, sharpe=0.1605, family=ridge, lag=0, epochs=120, patience=5, n_features=8
- Model_B_price_full_available: rmse=0.020379, mae=0.014013, dir_acc=0.5054, sharpe=0.2555, family=ridge, lag=0, epochs=120, patience=5, n_features=22
- Model_C_sentiment_only: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_D_price_shuffled_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_E_price_lagged_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
