# Panel Walk-Forward Experiment Summary

- Target: target_h5

## Feature Group Results
- G3_plus_panel_breadth: rmse=0.046401, mae=0.032966, dir_acc=0.5181, sharpe=0.2516, n_features=22
- G2_price_technical: rmse=0.046407, mae=0.032972, dir_acc=0.5174, sharpe=0.2236, n_features=17
- G1_price_only: rmse=0.046416, mae=0.032979, dir_acc=0.5160, sharpe=0.1821, n_features=8

## Model Family Sweep (Top 20 by RMSE)
- elastic_net | G3_plus_panel_breadth | lag=0 | epochs=300 | patience=20: rmse=0.046401, mae=0.032966, dir_acc=0.5181, sharpe=0.2516
- elastic_net | G2_price_technical | lag=0 | epochs=300 | patience=20: rmse=0.046407, mae=0.032972, dir_acc=0.5174, sharpe=0.2236
- elastic_net | G1_price_only | lag=0 | epochs=300 | patience=20: rmse=0.046416, mae=0.032979, dir_acc=0.5160, sharpe=0.1821
- ridge | G3_plus_panel_breadth | lag=0 | epochs=300 | patience=20: rmse=0.046424, mae=0.032993, dir_acc=0.5124, sharpe=0.1792
- ridge | G2_price_technical | lag=0 | epochs=300 | patience=20: rmse=0.046433, mae=0.033002, dir_acc=0.5137, sharpe=0.2060
- ridge | G1_price_only | lag=0 | epochs=300 | patience=20: rmse=0.046434, mae=0.032995, dir_acc=0.5112, sharpe=0.0817
- random_forest | G1_price_only | lag=0 | epochs=300 | patience=20: rmse=0.046532, mae=0.033029, dir_acc=0.5165, sharpe=0.1895
- hist_gbr | G1_price_only | lag=0 | epochs=300 | patience=20: rmse=0.046567, mae=0.033066, dir_acc=0.5141, sharpe=0.1330
- mlp | G1_price_only | lag=0 | epochs=300 | patience=20: rmse=0.046569, mae=0.033074, dir_acc=0.5161, sharpe=0.2236
- random_forest | G2_price_technical | lag=0 | epochs=300 | patience=20: rmse=0.046586, mae=0.033055, dir_acc=0.5147, sharpe=0.1247
- random_forest | G3_plus_panel_breadth | lag=0 | epochs=300 | patience=20: rmse=0.046768, mae=0.033177, dir_acc=0.5142, sharpe=-0.0071
- hist_gbr | G2_price_technical | lag=0 | epochs=300 | patience=20: rmse=0.046841, mae=0.033211, dir_acc=0.5069, sharpe=-0.0392
- mlp | G2_price_technical | lag=0 | epochs=300 | patience=20: rmse=0.046929, mae=0.033298, dir_acc=0.5075, sharpe=0.0857
- mlp | G3_plus_panel_breadth | lag=0 | epochs=300 | patience=20: rmse=0.046936, mae=0.033370, dir_acc=0.5119, sharpe=0.0884
- hist_gbr | G3_plus_panel_breadth | lag=0 | epochs=300 | patience=20: rmse=0.046986, mae=0.033305, dir_acc=0.5042, sharpe=-0.2418

## Model Set A-E
- Model_A_price_only: rmse=0.046416, mae=0.032979, dir_acc=0.5160, sharpe=0.1821, family=elastic_net, lag=0, epochs=300, patience=20, n_features=8
- Model_B_price_full_available: rmse=0.046401, mae=0.032966, dir_acc=0.5181, sharpe=0.2516, family=elastic_net, lag=0, epochs=300, patience=20, n_features=22
- Model_C_sentiment_only: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_D_price_shuffled_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_E_price_lagged_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
