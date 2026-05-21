"""
TOMLSignals - ML-Enhanced Signal Processing (Category 8)
==========================================================
1D CNN denoiser, LSTM denoiser, Small Transformer denoiser
"""

import torch
import torch.nn as nn
import numpy as np


# ---- 34. 1D CNN Denoiser ----

class CNN1DDenoiser(nn.Module):
    def __init__(self, channels=32, n_layers=3):
        super().__init__()
        layers = [nn.Conv1d(1, channels, 7, padding=3), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Conv1d(channels, channels, 7, padding=3), nn.ReLU()]
        layers += [nn.Conv1d(channels, 1, 7, padding=3)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def setup_cnn_denoiser(signal_length, batch_size, precision, device, channels=32, n_layers=3, **kw):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    model = CNN1DDenoiser(channels, n_layers).to(device).to(dtype).eval()
    x = torch.randn(batch_size, 1, signal_length, device=device, dtype=dtype)
    return {"model": model, "x": x}

def run_cnn_denoiser(state):
    with torch.no_grad():
        state["model"](state["x"])
    torch.cuda.synchronize()


# ---- 35. LSTM Denoiser ----

class LSTMDenoiser(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)


def setup_lstm_denoiser(signal_length, batch_size, precision, device, hidden_size=128, **kw):
    dtype = torch.float32  # LSTM doesn't support fp16 well on all GPUs
    model = LSTMDenoiser(hidden_size).to(device).to(dtype).eval()
    x = torch.randn(batch_size, signal_length, 1, device=device, dtype=dtype)
    return {"model": model, "x": x}

def run_lstm_denoiser(state):
    with torch.no_grad():
        state["model"](state["x"])
    torch.cuda.synchronize()


# ---- 36. Small Transformer Denoiser ----

class TransformerDenoiser(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=2, d_ff=128):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.encoder(h)
        return self.output_proj(h)


def setup_transformer_denoiser(signal_length, batch_size, precision, device, d_model=64, n_heads=4, n_layers=2, **kw):
    dtype = torch.float32
    model = TransformerDenoiser(d_model, n_heads, n_layers).to(device).to(dtype).eval()
    x = torch.randn(batch_size, signal_length, 1, device=device, dtype=dtype)
    return {"model": model, "x": x}

def run_transformer_denoiser(state):
    with torch.no_grad():
        state["model"](state["x"])
    torch.cuda.synchronize()


ML_ENHANCED = {
    "cnn_denoiser": (setup_cnn_denoiser, run_cnn_denoiser, {"channels": 32, "n_layers": 3}),
    "lstm_denoiser": (setup_lstm_denoiser, run_lstm_denoiser, {"hidden_size": 128}),
    "transformer_denoiser": (setup_transformer_denoiser, run_transformer_denoiser, {"d_model": 64, "n_heads": 4, "n_layers": 2}),
}
