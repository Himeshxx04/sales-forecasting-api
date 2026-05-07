import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


class _Net(nn.Module):
    def __init__(self, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMModel:
    name = "LSTM"

    def __init__(self, seq_len=30, hidden=64, epochs=60, lr=0.001):
        self.seq_len = seq_len
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.net = None
        self._seed_seq = None  # last seq_len scaled values, used to kick off prediction

    def _make_sequences(self, scaled):
        X, y = [], []
        for i in range(self.seq_len, len(scaled)):
            X.append(scaled[i - self.seq_len: i])
            y.append(scaled[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def fit(self, series):
        vals = series.values.reshape(-1, 1).astype(np.float64)
        scaled = self.scaler.fit_transform(vals).flatten()

        X, y = self._make_sequences(scaled)
        # shape: (samples, seq_len, 1)
        X_t = torch.from_numpy(X).unsqueeze(-1)
        y_t = torch.from_numpy(y).unsqueeze(-1)

        self.net = _Net(self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(self.net(X_t), y_t)
            loss.backward()
            opt.step()

        self._seed_seq = scaled[-self.seq_len:].tolist()
        self.net.eval()
        return self

    def predict(self, n_periods):
        assert self.net is not None, "call fit() first"
        history = self._seed_seq.copy()
        preds_scaled = []

        with torch.no_grad():
            for _ in range(n_periods):
                x = torch.tensor(
                    history[-self.seq_len:], dtype=torch.float32
                ).unsqueeze(0).unsqueeze(-1)
                p = self.net(x).item()
                preds_scaled.append(p)
                history.append(p)

        preds = self.scaler.inverse_transform(
            np.array(preds_scaled, dtype=np.float32).reshape(-1, 1)
        ).flatten()
        return np.maximum(preds, 0)
