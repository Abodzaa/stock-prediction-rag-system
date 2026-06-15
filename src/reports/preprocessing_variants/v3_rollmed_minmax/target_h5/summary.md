# Panel Walk-Forward Experiment Summary

- Target: target_h5

## Feature Group Results
- G2_price_technical: rmse=0.046382, mae=0.032994, dir_acc=0.5180, sharpe=0.5630, n_features=17
- G3_plus_panel_breadth: rmse=0.046390, mae=0.032998, dir_acc=0.5174, sharpe=0.5073, n_features=22
- G1_price_only: rmse=0.046425, mae=0.032997, dir_acc=0.5184, sharpe=0.2999, n_features=8

## Model Family Sweep (Top 20 by RMSE)
- elastic_net | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.046382, mae=0.032994, dir_acc=0.5180, sharpe=0.5630
- elastic_net | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.046390, mae=0.032998, dir_acc=0.5174, sharpe=0.5073
- ridge | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.046392, mae=0.033034, dir_acc=0.5119, sharpe=0.4703
- ridge | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.046417, mae=0.033052, dir_acc=0.5115, sharpe=0.3891
- elastic_net | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.046425, mae=0.032997, dir_acc=0.5184, sharpe=0.2999
- ridge | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.046434, mae=0.033008, dir_acc=0.5166, sharpe=0.2494
- hist_gbr | G1_price_only | lag=0 | epochs=120 | patience=5: rmse=0.046525, mae=0.033054, dir_acc=0.5171, sharpe=0.2023
- hist_gbr | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=5: rmse=0.046565, mae=0.033159, dir_acc=0.5191, sharpe=0.6801
- hist_gbr | G2_price_technical | lag=0 | epochs=120 | patience=5: rmse=0.046568, mae=0.033135, dir_acc=0.5137, sharpe=0.3679

## Model Set A-E
- Model_A_price_only: rmse=0.046425, mae=0.032997, dir_acc=0.5184, sharpe=0.2999, family=elastic_net, lag=0, epochs=120, patience=5, n_features=8
- Model_B_price_full_available: rmse=0.046390, mae=0.032998, dir_acc=0.5174, sharpe=0.5073, family=elastic_net, lag=0, epochs=120, patience=5, n_features=22
- Model_C_sentiment_only: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_D_price_shuffled_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
- Model_E_price_lagged_sentiment: not_run (No sentiment features with prefix g5_sent_ were found.)
