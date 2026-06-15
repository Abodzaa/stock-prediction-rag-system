# Sentiment Pipeline Comparison (H1 + H5)

- Selection: lowest average Model C RMSE across targets.

## Aggregate
- finbert: avg_c_rmse=0.012583, avg_c_dir_acc=0.6000, avg_c_sharpe=1.3125
- rag: avg_c_rmse=0.012583, avg_c_dir_acc=0.6000, avg_c_sharpe=1.3125

## Per Target
- h1 | finbert: c_rmse=0.008149, c_dir_acc=0.5667, c_sharpe=0.4987, delta_c_vs_d=0.000450, delta_c_vs_e=0.000450
- h1 | rag: c_rmse=0.008149, c_dir_acc=0.5667, c_sharpe=0.4987, delta_c_vs_d=0.000450, delta_c_vs_e=0.000450
- h5 | finbert: c_rmse=0.017016, c_dir_acc=0.6333, c_sharpe=2.1262, delta_c_vs_d=-0.000049, delta_c_vs_e=-0.000049
- h5 | rag: c_rmse=0.017016, c_dir_acc=0.6333, c_sharpe=2.1262, delta_c_vs_d=-0.000049, delta_c_vs_e=-0.000049

## Winner
- tie