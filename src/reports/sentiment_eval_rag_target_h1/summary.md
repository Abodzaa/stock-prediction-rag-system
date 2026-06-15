# Panel Walk-Forward Experiment Summary

- Target: target_h1

## Feature Group Results
- G3_plus_panel_breadth: rmse=0.020382, mae=0.014006, dir_acc=0.5116, sharpe=0.2609, n_features=22
- G1_price_only: rmse=0.020383, mae=0.014004, dir_acc=0.5109, sharpe=0.2083, n_features=8
- G2_price_technical: rmse=0.020385, mae=0.014006, dir_acc=0.5120, sharpe=0.1103, n_features=17

## Model Family Sweep (Top 20 by RMSE)
- hist_gbr | G3_plus_lagged_sentiment | lag=0 | epochs=120 | patience=20: rmse=0.020368, mae=0.014012, dir_acc=0.5049, sharpe=-0.1005
- elastic_net | G3_plus_shuffled_sentiment | lag=0 | epochs=300 | patience=20: rmse=0.020377, mae=0.014003, dir_acc=0.5123, sharpe=0.2068
- ridge | G3_plus_panel_breadth | lag=0 | epochs=300 | patience=20: rmse=0.020382, mae=0.014006, dir_acc=0.5116, sharpe=0.2609
- ridge | G1_price_only | lag=0 | epochs=300 | patience=20: rmse=0.020383, mae=0.014004, dir_acc=0.5109, sharpe=0.2083
- elastic_net | G2_price_technical | lag=0 | epochs=300 | patience=20: rmse=0.020385, mae=0.014006, dir_acc=0.5120, sharpe=0.1103
- elastic_net | G1_price_only | lag=0 | epochs=300 | patience=20: rmse=0.020385, mae=0.014006, dir_acc=0.5121, sharpe=0.1103
- elastic_net | G3_plus_panel_breadth | lag=0 | epochs=300 | patience=20: rmse=0.020385, mae=0.014006, dir_acc=0.5119, sharpe=0.1083
- ridge | G3_plus_shuffled_sentiment | lag=0 | epochs=300 | patience=20: rmse=0.020386, mae=0.014011, dir_acc=0.5112, sharpe=0.2236
- ridge | G2_price_technical | lag=0 | epochs=300 | patience=20: rmse=0.020386, mae=0.014010, dir_acc=0.5092, sharpe=0.1881
- elastic_net | G5_sentiment_only | lag=0 | epochs=300 | patience=20: rmse=0.020388, mae=0.014010, dir_acc=0.5125, sharpe=0.1418
- ridge | G5_sentiment_only | lag=0 | epochs=300 | patience=20: rmse=0.020388, mae=0.014010, dir_acc=0.5125, sharpe=0.1418
- hist_gbr | G5_sentiment_only | lag=0 | epochs=120 | patience=20: rmse=0.020388, mae=0.014010, dir_acc=0.5125, sharpe=0.1418
- elastic_net | G3_plus_lagged_sentiment | lag=0 | epochs=300 | patience=20: rmse=0.020388, mae=0.014006, dir_acc=0.5118, sharpe=0.1232
- ridge | G3_plus_lagged_sentiment | lag=0 | epochs=300 | patience=20: rmse=0.020388, mae=0.014009, dir_acc=0.5106, sharpe=0.1893
- hist_gbr | G3_plus_panel_breadth | lag=0 | epochs=120 | patience=20: rmse=0.020392, mae=0.014019, dir_acc=0.5081, sharpe=-0.0156
- hist_gbr | G3_plus_shuffled_sentiment | lag=0 | epochs=120 | patience=20: rmse=0.020451, mae=0.014021, dir_acc=0.5095, sharpe=-0.0904
- hist_gbr | G1_price_only | lag=0 | epochs=120 | patience=20: rmse=0.020455, mae=0.014032, dir_acc=0.5073, sharpe=-0.0449
- hist_gbr | G2_price_technical | lag=0 | epochs=120 | patience=20: rmse=0.020497, mae=0.014048, dir_acc=0.5090, sharpe=-0.0272

## Model Set A-E
- Model_A_price_only: rmse=0.020383, mae=0.014004, dir_acc=0.5109, sharpe=0.2083, family=ridge, lag=0, epochs=300, patience=20, n_features=8
- Model_B_price_full_available: rmse=0.020382, mae=0.014006, dir_acc=0.5116, sharpe=0.2609, family=ridge, lag=0, epochs=300, patience=20, n_features=22
- Model_C_sentiment_only: rmse=0.020388, mae=0.014010, dir_acc=0.5125, sharpe=0.1418, family=elastic_net, lag=0, epochs=300, patience=20, n_features=14
- Model_D_price_shuffled_sentiment: rmse=0.020377, mae=0.014003, dir_acc=0.5123, sharpe=0.2068, family=elastic_net, lag=0, epochs=300, patience=20, n_features=36
- Model_E_price_lagged_sentiment: rmse=0.020368, mae=0.014012, dir_acc=0.5049, sharpe=-0.1005, family=hist_gbr, lag=0, epochs=120, patience=20, n_features=36
