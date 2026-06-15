# Walk-Forward Experiment Summary

- Target: target_h1

## Feature Group Results
- G1_price_only: rmse=0.010595, mae=0.007550, dir_acc=0.5373, sharpe=0.5955, n_features=248
- G2_price_technical: rmse=0.010641, mae=0.007582, dir_acc=0.5310, sharpe=0.7941, n_features=187
- G3_plus_breadth: rmse=0.010669, mae=0.007610, dir_acc=0.5310, sharpe=0.6062, n_features=264

## Model Family Sweep (Top 12 by RMSE)
- random_forest | G1_price_only | lag=30 | epochs=300 | patience=20: rmse=0.010595, mae=0.007550, dir_acc=0.5373, sharpe=0.5955
- random_forest | G1_price_only | lag=10 | epochs=300 | patience=20: rmse=0.010637, mae=0.007553, dir_acc=0.5365, sharpe=0.6988
- hist_gbr | G2_price_technical | lag=10 | epochs=200 | patience=20: rmse=0.010641, mae=0.007582, dir_acc=0.5310, sharpe=0.7941
- hist_gbr | G2_price_technical | lag=10 | epochs=350 | patience=20: rmse=0.010641, mae=0.007582, dir_acc=0.5310, sharpe=0.7941
- random_forest | G2_price_technical | lag=30 | epochs=300 | patience=20: rmse=0.010653, mae=0.007606, dir_acc=0.5317, sharpe=0.6047
- hist_gbr | G2_price_technical | lag=10 | epochs=200 | patience=10: rmse=0.010662, mae=0.007607, dir_acc=0.5333, sharpe=0.7065
- hist_gbr | G2_price_technical | lag=10 | epochs=350 | patience=10: rmse=0.010662, mae=0.007607, dir_acc=0.5333, sharpe=0.7065
- hist_gbr | G2_price_technical | lag=30 | epochs=200 | patience=10: rmse=0.010663, mae=0.007603, dir_acc=0.5357, sharpe=0.9002
- hist_gbr | G2_price_technical | lag=30 | epochs=350 | patience=10: rmse=0.010663, mae=0.007603, dir_acc=0.5357, sharpe=0.9002
- hist_gbr | G3_plus_breadth | lag=10 | epochs=350 | patience=20: rmse=0.010669, mae=0.007610, dir_acc=0.5310, sharpe=0.6062
- hist_gbr | G3_plus_breadth | lag=10 | epochs=200 | patience=20: rmse=0.010669, mae=0.007610, dir_acc=0.5310, sharpe=0.6062
- hist_gbr | G1_price_only | lag=10 | epochs=350 | patience=20: rmse=0.010678, mae=0.007600, dir_acc=0.5373, sharpe=0.5018

## Model Set A-E Results
- Model_A_price_only: rmse=0.010595, mae=0.007550, dir_acc=0.5373, sharpe=0.5955, n_features=248, model_family=random_forest, lag=30, epochs=300, patience=20
- Model_B_price_full_available: rmse=0.010669, mae=0.007610, dir_acc=0.5310, sharpe=0.6062, n_features=264, model_family=hist_gbr, lag=10, epochs=200, patience=20
- Model_C_sentiment_only: not_run (No sentiment features with prefix g5_sent_ were found in features CSV.)
- Model_D_price_shuffled_sentiment: not_run (No sentiment features with prefix g5_sent_ were found in features CSV.)
- Model_E_price_lagged_sentiment: not_run (No sentiment features with prefix g5_sent_ were found in features CSV.)
