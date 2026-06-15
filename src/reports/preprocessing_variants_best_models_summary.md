# Best Models on Preprocessing Variants

Settings: models=elastic_net,ridge,hist_gbr | lag=0 | epochs=120 | patience=5

## target_h1
- v2_ema_standardz: rmse=0.020322460, dir_acc=0.508581, model_family=hist_gbr, group=G3_plus_panel_breadth
- v4_hampel_ranknorm: rmse=0.020371563, dir_acc=0.516804, model_family=ridge, group=G1_price_only
- v3_rollmed_minmax: rmse=0.020378797, dir_acc=0.505357, model_family=ridge, group=G3_plus_panel_breadth
- v1_winsor_robustz: rmse=0.020383422, dir_acc=0.513638, model_family=elastic_net, group=G2_price_technical
- v5_iqrclip_log_cs_z: rmse=0.020386987, dir_acc=0.512539, model_family=elastic_net, group=G1_price_only

## target_h5
- v3_rollmed_minmax: rmse=0.046381820, dir_acc=0.517978, model_family=elastic_net, group=G2_price_technical
- v4_hampel_ranknorm: rmse=0.046382420, dir_acc=0.522020, model_family=ridge, group=G1_price_only
- v2_ema_standardz: rmse=0.046403910, dir_acc=0.513718, model_family=elastic_net, group=G3_plus_panel_breadth
- v1_winsor_robustz: rmse=0.046411403, dir_acc=0.517006, model_family=elastic_net, group=G1_price_only
- v5_iqrclip_log_cs_z: rmse=0.046417852, dir_acc=0.518737, model_family=hist_gbr, group=G2_price_technical
