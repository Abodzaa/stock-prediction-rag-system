from __future__ import annotations

from torch import nn

from src.models.autoformer_model import AutoformerRegressor
from src.models.fedformer_model import FEDformerRegressor
from src.models.gru_model import GRURegressor
from src.models.informer_model import InformerRegressor
from src.models.lstm_model import LSTMRegressor
from src.models.mamba_stock_model import MambaStockRegressor
from src.models.nbeats_model import NBEATSRegressor
from src.models.nhits_model import NHITSRegressor
from src.models.patchtst_model import PatchTSTRegressor
from src.models.tft_model import TFTRegressor


def build_torch_model(model_name: str, input_size: int, seq_len: int, hidden_size: int) -> nn.Module:
    if model_name == "LSTM":
        return LSTMRegressor(input_size=input_size, hidden_size=hidden_size)
    if model_name == "GRU":
        return GRURegressor(input_size=input_size, hidden_size=hidden_size)
    if model_name == "NBEATS":
        return NBEATSRegressor(input_size=input_size, seq_len=seq_len, hidden_size=hidden_size)
    if model_name == "NHITS":
        return NHITSRegressor(hidden_size=hidden_size)
    if model_name == "PatchTST":
        return PatchTSTRegressor(input_size=input_size, hidden_size=hidden_size)
    if model_name == "Informer":
        return InformerRegressor(input_size=input_size, hidden_size=hidden_size)
    if model_name == "Autoformer":
        return AutoformerRegressor(input_size=input_size, hidden_size=hidden_size)
    if model_name == "FEDformer":
        return FEDformerRegressor(input_size=input_size, hidden_size=hidden_size)
    if model_name == "TFT":
        return TFTRegressor(input_size=input_size, hidden_size=hidden_size)
    if model_name in {"MambaStock", "Mamba"}:
        return MambaStockRegressor(input_size=input_size, hidden_size=hidden_size)

    raise ValueError(f"Unsupported model_name: {model_name}")
