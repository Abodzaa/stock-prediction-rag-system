# Walk-Forward Experiment Summary

- Target: target_h5

## Feature Group Results
- G1_price_only: rmse=0.022703, mae=0.016901, dir_acc=0.6032, sharpe=1.8304, n_features=88
- G2_price_technical: rmse=0.023613, mae=0.017601, dir_acc=0.5532, sharpe=1.2442, n_features=187
- G3_plus_breadth: rmse=0.024565, mae=0.017848, dir_acc=0.5262, sharpe=0.3627, n_features=264

## Model Family Sweep (Top 12 by RMSE)
- hist_gbr | G1_price_only | lag=10 | epochs=200 | patience=10: rmse=0.022703, mae=0.016901, dir_acc=0.6032, sharpe=1.8304
- hist_gbr | G1_price_only | lag=10 | epochs=350 | patience=10: rmse=0.022703, mae=0.016901, dir_acc=0.6032, sharpe=1.8304
- hist_gbr | G1_price_only | lag=30 | epochs=350 | patience=10: rmse=0.022711, mae=0.016813, dir_acc=0.6016, sharpe=1.6843
- hist_gbr | G1_price_only | lag=30 | epochs=200 | patience=10: rmse=0.022711, mae=0.016813, dir_acc=0.6016, sharpe=1.6843
- hist_gbr | G1_price_only | lag=10 | epochs=200 | patience=20: rmse=0.022716, mae=0.016910, dir_acc=0.5937, sharpe=1.6563
- hist_gbr | G1_price_only | lag=10 | epochs=350 | patience=20: rmse=0.022716, mae=0.016910, dir_acc=0.5937, sharpe=1.6563
- hist_gbr | G1_price_only | lag=30 | epochs=350 | patience=20: rmse=0.022753, mae=0.016855, dir_acc=0.5952, sharpe=1.6415
- hist_gbr | G1_price_only | lag=30 | epochs=200 | patience=20: rmse=0.022753, mae=0.016855, dir_acc=0.5952, sharpe=1.6415
- random_forest | G1_price_only | lag=10 | epochs=300 | patience=20: rmse=0.022947, mae=0.016917, dir_acc=0.5913, sharpe=1.2092
- elastic_net | G1_price_only | lag=10 | epochs=300 | patience=20: rmse=0.022987, mae=0.017076, dir_acc=0.5508, sharpe=1.2759
- random_forest | G1_price_only | lag=30 | epochs=300 | patience=20: rmse=0.023001, mae=0.016918, dir_acc=0.5865, sharpe=0.7851
- ridge | G1_price_only | lag=10 | epochs=300 | patience=20: rmse=0.023419, mae=0.017268, dir_acc=0.5571, sharpe=1.4463

## Model Set A-E Results
- Model_A_price_only: rmse=0.022703, mae=0.016901, dir_acc=0.6032, sharpe=1.8304, n_features=88, model_family=hist_gbr, lag=10, epochs=200, patience=10
- Model_B_price_full_available: rmse=0.024565, mae=0.017848, dir_acc=0.5262, sharpe=0.3627, n_features=264, model_family=elastic_net, lag=10, epochs=300, patience=20
- Model_C_sentiment_only: not_run (No sentiment features with prefix g5_sent_ were found in features CSV.)
- Model_D_price_shuffled_sentiment: not_run (No sentiment features with prefix g5_sent_ were found in features CSV.)
- Model_E_price_lagged_sentiment: not_run (No sentiment features with prefix g5_sent_ were found in features CSV.)
