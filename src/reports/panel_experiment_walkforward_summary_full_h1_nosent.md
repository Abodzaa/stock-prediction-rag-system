# Panel Walk-Forward Experiment Summary

- Target: target_h1

## Feature Group Results
- G3_plus_panel_breadth: rmse=0.020374, mae=0.014002, dir_acc=0.5127, sharpe=0.2480, n_features=22
- G1_price_only: rmse=0.020377, mae=0.014003, dir_acc=0.5105, sharpe=0.1685, n_features=8
- G2_price_technical: rmse=0.020379, mae=0.014006, dir_acc=0.5101, sharpe=0.1932, n_features=17

## Model Family Sweep (Top 20 by RMSE)
- ridge | G3_plus_panel_breadth | lag=0 | epochs=300 | patience=20: rmse=0.020374, mae=0.014002, dir_acc=0.5127, sharpe=0.2480
- ridge | G1_price_only | lag=0 | epochs=300 | patience=20: rmse=0.020377, mae=0.014003, dir_acc=0.5105, sharpe=0.1685
- ridge | G2_price_technical | lag=0 | epochs=300 | patience=20: rmse=0.020379, mae=0.014006, dir_acc=0.5101, sharpe=0.1932
- elastic_net | G2_price_technical | lag=0 | epochs=300 | patience=20: rmse=0.020379, mae=0.014004, dir_acc=0.5123, sharpe=0.1442
- elastic_net | G1_price_only | lag=0 | epochs=300 | patience=20: rmse=0.020379, mae=0.014004, dir_acc=0.5123, sharpe=0.1442
- elastic_net | G3_plus_panel_breadth | lag=0 | epochs=300 | patience=20: rmse=0.020379, mae=0.014005, dir_acc=0.5125, sharpe=0.1515
- hist_gbr | G1_price_only | lag=0 | epochs=300 | patience=20: rmse=0.020419, mae=0.014021, dir_acc=0.5098, sharpe=0.0675
- random_forest | G1_price_only | lag=0 | epochs=300 | patience=20: rmse=0.020439, mae=0.014024, dir_acc=0.5096, sharpe=-0.0017
- random_forest | G2_price_technical | lag=0 | epochs=300 | patience=20: rmse=0.020441, mae=0.014027, dir_acc=0.5093, sharpe=-0.0470
- random_forest | G3_plus_panel_breadth | lag=0 | epochs=300 | patience=20: rmse=0.020449, mae=0.014052, dir_acc=0.5085, sharpe=-0.0125
- hist_gbr | G3_plus_panel_breadth | lag=0 | epochs=300 | patience=20: rmse=0.020456, mae=0.014062, dir_acc=0.5042, sharpe=-0.0875
- mlp | G3_plus_panel_breadth | lag=0 | epochs=300 | patience=20: rmse=0.020476, mae=0.014077, dir_acc=0.5061, sharpe=0.2026
- mlp | G2_price_technical | lag=0 | epochs=300 | patience=20: rmse=0.020479, mae=0.014054, dir_acc=0.5098, sharpe=0.1289
- mlp | G1_price_only | lag=0 | epochs=300 | patience=20: rmse=0.020497, mae=0.014042, dir_acc=0.5082, sharpe=-0.1140
- hist_gbr | G2_price_technical | lag=0 | epochs=300 | patience=20: rmse=0.020508, mae=0.014052, dir_acc=0.5073, sharpe=-0.0558

## Model Set A-E
- Model_A_price_only: rmse=0.020377, mae=0.014003, dir_acc=0.5105, sharpe=0.1685, family=ridge, lag=0, epochs=300, patience=20, n_features=8
- Model_B_price_full_available: rmse=0.020374, mae=0.014002, dir_acc=0.5127, sharpe=0.2480, family=ridge, lag=0, epochs=300, patience=20, n_features=22
- Model_C_sentiment_only: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_D_price_shuffled_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_E_price_lagged_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
