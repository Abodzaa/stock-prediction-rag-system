# Panel Walk-Forward Experiment Summary

- Target: target_h5

## Feature Group Results
- G3_plus_panel_breadth: rmse=0.046399, mae=0.032967, dir_acc=0.5168, sharpe=0.2184, n_features=22
- G2_price_technical: rmse=0.046404, mae=0.032973, dir_acc=0.5164, sharpe=0.2114, n_features=17
- G1_price_only: rmse=0.046416, mae=0.032980, dir_acc=0.5153, sharpe=0.1512, n_features=8

## Model Family Sweep (Top 20 by RMSE)
- elastic_net | G3_plus_panel_breadth | lag=0 | epochs=300 | patience=20: rmse=0.046399, mae=0.032967, dir_acc=0.5168, sharpe=0.2184
- elastic_net | G2_price_technical | lag=0 | epochs=300 | patience=20: rmse=0.046404, mae=0.032973, dir_acc=0.5164, sharpe=0.2114
- elastic_net | G1_price_only | lag=0 | epochs=300 | patience=20: rmse=0.046416, mae=0.032980, dir_acc=0.5153, sharpe=0.1512
- ridge | G3_plus_panel_breadth | lag=0 | epochs=300 | patience=20: rmse=0.046416, mae=0.032993, dir_acc=0.5121, sharpe=0.1920
- elastic_net | G3_plus_lagged_sentiment | lag=0 | epochs=300 | patience=20: rmse=0.046424, mae=0.032982, dir_acc=0.5167, sharpe=0.2354
- ridge | G2_price_technical | lag=0 | epochs=300 | patience=20: rmse=0.046426, mae=0.033003, dir_acc=0.5130, sharpe=0.2119
- elastic_net | G5_sentiment_only | lag=0 | epochs=300 | patience=20: rmse=0.046426, mae=0.033000, dir_acc=0.5200, sharpe=0.3106
- ridge | G5_sentiment_only | lag=0 | epochs=300 | patience=20: rmse=0.046426, mae=0.033000, dir_acc=0.5200, sharpe=0.3106
- ridge | G1_price_only | lag=0 | epochs=300 | patience=20: rmse=0.046426, mae=0.032993, dir_acc=0.5111, sharpe=0.0870
- hist_gbr | G5_sentiment_only | lag=0 | epochs=150 | patience=20: rmse=0.046426, mae=0.033000, dir_acc=0.5200, sharpe=0.3106
- elastic_net | G3_plus_shuffled_sentiment | lag=0 | epochs=300 | patience=20: rmse=0.046427, mae=0.032983, dir_acc=0.5170, sharpe=0.2294
- ridge | G3_plus_lagged_sentiment | lag=0 | epochs=300 | patience=20: rmse=0.046457, mae=0.033016, dir_acc=0.5114, sharpe=0.1689
- ridge | G3_plus_shuffled_sentiment | lag=0 | epochs=300 | patience=20: rmse=0.046457, mae=0.033012, dir_acc=0.5133, sharpe=0.2039
- hist_gbr | G1_price_only | lag=0 | epochs=150 | patience=20: rmse=0.046494, mae=0.033029, dir_acc=0.5162, sharpe=0.1999
- hist_gbr | G2_price_technical | lag=0 | epochs=150 | patience=20: rmse=0.046666, mae=0.033126, dir_acc=0.5112, sharpe=0.1039
- hist_gbr | G3_plus_panel_breadth | lag=0 | epochs=150 | patience=20: rmse=0.046721, mae=0.033154, dir_acc=0.5071, sharpe=-0.2023
- hist_gbr | G3_plus_shuffled_sentiment | lag=0 | epochs=150 | patience=20: rmse=0.046763, mae=0.033187, dir_acc=0.5094, sharpe=-0.0306
- hist_gbr | G3_plus_lagged_sentiment | lag=0 | epochs=150 | patience=20: rmse=0.046801, mae=0.033169, dir_acc=0.5128, sharpe=0.0448

## Model Set A-E
- Model_A_price_only: rmse=0.046416, mae=0.032980, dir_acc=0.5153, sharpe=0.1512, family=elastic_net, lag=0, epochs=300, patience=20, n_features=8
- Model_B_price_full_available: rmse=0.046399, mae=0.032967, dir_acc=0.5168, sharpe=0.2184, family=elastic_net, lag=0, epochs=300, patience=20, n_features=22
- Model_C_sentiment_only: rmse=0.046426, mae=0.033000, dir_acc=0.5200, sharpe=0.3106, family=elastic_net, lag=0, epochs=300, patience=20, n_features=14
- Model_D_price_shuffled_sentiment: rmse=0.046427, mae=0.032983, dir_acc=0.5170, sharpe=0.2294, family=elastic_net, lag=0, epochs=300, patience=20, n_features=36
- Model_E_price_lagged_sentiment: rmse=0.046424, mae=0.032982, dir_acc=0.5167, sharpe=0.2354, family=elastic_net, lag=0, epochs=300, patience=20, n_features=36
