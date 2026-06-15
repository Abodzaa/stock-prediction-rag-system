# NASDAQ-100 Best Deep Models Per Symbol

Selection source: S&P-derived best configs from thesis_best_model_summary.csv.
Ranking: lowest RMSE, tie-break by higher directional accuracy then higher Sharpe.

## target_h1
- EA | PatchTST | G1_price_only: rmse=0.009254, dir_acc=0.5397, sharpe=1.8387
- EA | TFT | G2_price_technical: rmse=0.009444, dir_acc=0.4444, sharpe=-2.3334
- EA | LSTM | G3_plus_breadth: rmse=0.010122, dir_acc=0.5000, sharpe=-0.5893
- AEP | Informer | G2_price_technical: rmse=0.010937, dir_acc=0.5159, sharpe=0.8126
- AEP | TFT | G2_price_technical: rmse=0.011034, dir_acc=0.6429, sharpe=2.7455
- EA | Informer | G2_price_technical: rmse=0.011090, dir_acc=0.4921, sharpe=-1.4871
- AEP | PatchTST | G1_price_only: rmse=0.011134, dir_acc=0.5635, sharpe=0.4287
- AEP | GRU | G3_plus_breadth: rmse=0.011257, dir_acc=0.5714, sharpe=1.8289
- EXC | LSTM | G3_plus_breadth: rmse=0.011466, dir_acc=0.5159, sharpe=0.2983
- AEP | LSTM | G3_plus_breadth: rmse=0.011479, dir_acc=0.5714, sharpe=2.0382

## target_h5
- EA | TFT | G3_plus_breadth: rmse=0.017502, dir_acc=0.5476, sharpe=1.6991
- EA | LSTM | G3_plus_breadth: rmse=0.018614, dir_acc=0.5000, sharpe=-1.7426
- EA | GRU | G3_plus_breadth: rmse=0.019129, dir_acc=0.4921, sharpe=-2.4603
- EA | PatchTST | G1_price_only: rmse=0.019245, dir_acc=0.5952, sharpe=1.9751
- AEP | FEDformer | G1_price_only: rmse=0.024041, dir_acc=0.7063, sharpe=6.9431
- AEP | PatchTST | G1_price_only: rmse=0.024678, dir_acc=0.6429, sharpe=5.6529
- AEP | GRU | G3_plus_breadth: rmse=0.024711, dir_acc=0.5873, sharpe=4.2466
- AEP | TFT | G3_plus_breadth: rmse=0.024933, dir_acc=0.6587, sharpe=4.6459
- AEP | LSTM | G3_plus_breadth: rmse=0.025177, dir_acc=0.6111, sharpe=3.3127
- AEP | Autoformer | G2_price_technical: rmse=0.025236, dir_acc=0.6270, sharpe=2.7130
