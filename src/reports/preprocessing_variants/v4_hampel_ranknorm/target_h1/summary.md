# Panel Walk-Forward Experiment Summary

- Target: target_h1

## Feature Group Results
- G1_price_only: rmse=0.020372, mae=0.014000, dir_acc=0.5168, sharpe=0.5366, n_features=8
- G3_plus_panel_breadth: rmse=0.020372, mae=0.014004, dir_acc=0.5162, sharpe=0.5524, n_features=22
- G2_price_technical: rmse=0.020373, mae=0.014003, dir_acc=0.5163, sharpe=0.5363, n_features=17

## Model Family Sweep (Top 20 by RMSE)
- ridge | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.020372, mae=0.014000, dir_acc=0.5168, sharpe=0.5366
- ridge | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.020372, mae=0.014004, dir_acc=0.5162, sharpe=0.5524
- ridge | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.020373, mae=0.014003, dir_acc=0.5163, sharpe=0.5363
- hist_gbr | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.020375, mae=0.014003, dir_acc=0.5156, sharpe=0.3132
- hist_gbr | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.020377, mae=0.014004, dir_acc=0.5162, sharpe=0.3702
- elastic_net | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.020380, mae=0.014006, dir_acc=0.5127, sharpe=0.1623
- elastic_net | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.020380, mae=0.014006, dir_acc=0.5126, sharpe=0.1577
- elastic_net | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.020380, mae=0.014006, dir_acc=0.5126, sharpe=0.1577
- hist_gbr | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.020383, mae=0.014007, dir_acc=0.5121, sharpe=0.1272

## Model Set A-E
- Model_A_price_only: rmse=0.020372, mae=0.014000, dir_acc=0.5168, sharpe=0.5366, family=ridge, lag=0, epochs=120, patience=5, n_features=8
- Model_B_price_full_available: rmse=0.020372, mae=0.014004, dir_acc=0.5162, sharpe=0.5524, family=ridge, lag=0, epochs=120, patience=5, n_features=22
- Model_C_sentiment_only: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_D_price_shuffled_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_E_price_lagged_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
