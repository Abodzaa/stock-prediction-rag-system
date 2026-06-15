# Panel Walk-Forward Experiment Summary

- Target: target_h1

## Feature Group Results
- G2_price_technical: rmse=0.020383, mae=0.014006, dir_acc=0.5136, sharpe=0.1701, n_features=17
- G1_price_only: rmse=0.020384, mae=0.014006, dir_acc=0.5133, sharpe=0.1530, n_features=8
- G3_plus_panel_breadth: rmse=0.020384, mae=0.014006, dir_acc=0.5135, sharpe=0.1816, n_features=22

## Model Family Sweep (Top 20 by RMSE)
- elastic_net | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.020383, mae=0.014006, dir_acc=0.5136, sharpe=0.1701
- elastic_net | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.020384, mae=0.014006, dir_acc=0.5133, sharpe=0.1530
- elastic_net | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.020384, mae=0.014006, dir_acc=0.5135, sharpe=0.1816
- hist_gbr | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.020386, mae=0.014034, dir_acc=0.5052, sharpe=0.0194
- ridge | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.020396, mae=0.014008, dir_acc=0.5091, sharpe=0.0628
- ridge | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.020400, mae=0.014018, dir_acc=0.5102, sharpe=0.2211
- ridge | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.020403, mae=0.014020, dir_acc=0.5070, sharpe=0.1167
- hist_gbr | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.020408, mae=0.014016, dir_acc=0.5094, sharpe=-0.0154
- hist_gbr | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.020428, mae=0.014027, dir_acc=0.5097, sharpe=0.0023

## Model Set A-E
- Model_A_price_only: rmse=0.020384, mae=0.014006, dir_acc=0.5133, sharpe=0.1530, family=elastic_net, lag=0, epochs=120, patience=5, n_features=8
- Model_B_price_full_available: rmse=0.020384, mae=0.014006, dir_acc=0.5135, sharpe=0.1816, family=elastic_net, lag=0, epochs=120, patience=5, n_features=22
- Model_C_sentiment_only: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_D_price_shuffled_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_E_price_lagged_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
