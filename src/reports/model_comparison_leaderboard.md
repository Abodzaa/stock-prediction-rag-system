# Model Comparison Leaderboard

## Deep Best Overall Per Target
- h1: Informer | G2_price_technical | rmse=0.007739 | dir_acc=0.5714 | sharpe=3.1570
- h5: FEDformer | G1_price_only | rmse=0.015699 | dir_acc=0.6349 | sharpe=5.2681

## Deep Best Per Model Per Target (RMSE)
### h1
- Informer: rmse=0.007739, dir_acc=0.5714, sharpe=3.1570, fg=G2_price_technical
- TFT: rmse=0.007805, dir_acc=0.5714, sharpe=0.5232, fg=G2_price_technical
- PatchTST: rmse=0.007876, dir_acc=0.4921, sharpe=2.0258, fg=G1_price_only
- GRU: rmse=0.008007, dir_acc=0.6508, sharpe=3.8428, fg=G3_plus_breadth
- FEDformer: rmse=0.008216, dir_acc=0.5238, sharpe=1.1431, fg=G3_plus_breadth
- NHITS: rmse=0.008393, dir_acc=0.5397, sharpe=0.9271, fg=G3_plus_breadth
- MambaStock: rmse=0.008930, dir_acc=0.4444, sharpe=-1.2952, fg=G3_plus_breadth
- LSTM: rmse=0.011439, dir_acc=0.5873, sharpe=0.6219, fg=G3_plus_breadth
- Autoformer: rmse=0.011640, dir_acc=0.4444, sharpe=0.5365, fg=G2_price_technical
- NBEATS: rmse=0.036602, dir_acc=0.4444, sharpe=-1.5061, fg=G1_price_only
### h5
- FEDformer: rmse=0.015699, dir_acc=0.6349, sharpe=5.2681, fg=G1_price_only
- PatchTST: rmse=0.016168, dir_acc=0.6508, sharpe=2.6397, fg=G1_price_only
- Informer: rmse=0.016276, dir_acc=0.7302, sharpe=5.2635, fg=G2_price_technical
- GRU: rmse=0.016602, dir_acc=0.6190, sharpe=4.2228, fg=G3_plus_breadth
- TFT: rmse=0.016802, dir_acc=0.6508, sharpe=2.6397, fg=G3_plus_breadth
- MambaStock: rmse=0.017870, dir_acc=0.5714, sharpe=1.6311, fg=G2_price_technical
- LSTM: rmse=0.018711, dir_acc=0.4762, sharpe=-2.6060, fg=G3_plus_breadth
- NHITS: rmse=0.019042, dir_acc=0.5873, sharpe=4.0371, fg=G2_price_technical
- Autoformer: rmse=0.020383, dir_acc=0.3810, sharpe=-0.4944, fg=G2_price_technical
- NBEATS: rmse=0.034849, dir_acc=0.5556, sharpe=1.5855, fg=G1_price_only

## Classical Baseline Best Per Target
- h1: Model_A_price_only | rmse=0.010595 | dir_acc=0.5373 | sharpe=0.5955
- h5: Model_A_price_only | rmse=0.022703 | dir_acc=0.6032 | sharpe=1.8304