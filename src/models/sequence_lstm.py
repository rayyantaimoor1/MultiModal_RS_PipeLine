"""
sequence_lstm.py

Stacked recurrent network (LSTM or GRU) for multi-step forecasting of
vegetation-index time series (e.g. NDVI/EVI derived from multi-year
Sentinel-2 or MODIS composites). Consumes a lookback window of per-timestep
feature vectors and directly outputs a multi-step-ahead forecast vector
(sequence-to-vector, "direct multi-horizon" strategy — avoids compounding
autoregressive error).

Author: Principal ML Engineering Team
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
from torch import nn


class NDVILSTMForecaster(nn.Module):
    """Stacked LSTM/GRU forecaster for multi-step vegetation-index prediction.

    Encodes an input sequence with a stacked recurrent network, takes the
    final hidden state (top layer) as a fixed-size sequence summary, and
    projects it through a small dense head to directly output all
    `horizon` future steps at once.

    Args:
        input_dim: Number of features per timestep, F (e.g. 1 for NDVI-only,
            or >1 for [NDVI, EVI, LST, precip, ...]).
        hidden_dim: Hidden state size of each recurrent layer.
        num_layers: Number of stacked recurrent layers.
        horizon: Number of future timesteps to forecast (multi-step output).
        rnn_type: Either "lstm" or "gru".
        dropout_prob: Dropout probability between stacked recurrent layers
            (only applied when `num_layers > 1`, matching PyTorch's RNN
            dropout semantics) and in the final dense head.
        bidirectional: Whether the recurrent encoder is bidirectional. Note
            that for *forecasting* (as opposed to sequence labeling),
            bidirectional encoding of the lookback window is still valid
            since all lookback steps are already observed at inference
            time; only future steps are unknown.

    Shape:
        - Input: `[batch_size, lookback, input_dim]`
        - Output: `[batch_size, horizon]` (forecast of the target feature,
          e.g. NDVI, for each of the next `horizon` timesteps)

    Example:
        >>> model = NDVILSTMForecaster(input_dim=2, hidden_dim=64, num_layers=2, horizon=4)
        >>> x = torch.randn(16, 12, 2)  # [batch, lookback, features]
        >>> forecast = model(x)
        >>> forecast.shape
        torch.Size([16, 4])
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        horizon: int = 4,
        rnn_type: Literal["lstm", "gru"] = "lstm",
        dropout_prob: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        recurrent_dropout = dropout_prob if num_layers > 1 else 0.0

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )

        direction_multiplier = 2 if bidirectional else 1
        encoder_output_dim = hidden_dim * direction_multiplier

        self.forecast_head = nn.Sequential(
            nn.Linear(encoder_output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(
        self, x: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: FloatTensor, shape [batch_size, lookback, input_dim]
            hidden_state: Optional initial hidden state to pass to the RNN
                (e.g. for stateful rollouts across successive calls). For
                LSTM this is a tuple `(h_0, c_0)`; for GRU it is `h_0`
                directly. If `None`, PyTorch initializes it to zeros.

        Returns:
            FloatTensor, shape [batch_size, horizon] — the direct
            multi-step-ahead forecast of the primary target feature.
        """
        # x: [B, lookback, input_dim]
        rnn_output, _final_hidden = self.rnn(x, hidden_state)
        # rnn_output: [B, lookback, hidden_dim * num_directions]

        # Take the last timestep's output as the fixed-size sequence summary.
        last_step_output = rnn_output[:, -1, :]  # [B, hidden_dim * num_directions]

        forecast = self.forecast_head(last_step_output)  # [B, horizon]
        return forecast

    def forward_with_embedding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that also returns the sequence embedding.

        Useful for downstream tasks (e.g. anomaly scoring, clustering of
        seasonal vegetation trajectories) that need the learned
        fixed-length representation of the input window.

        Args:
            x: FloatTensor, shape [batch_size, lookback, input_dim]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                forecast: FloatTensor, shape [batch_size, horizon]
                embedding: FloatTensor, shape [batch_size, hidden_dim * num_directions]
        """
        rnn_output, _ = self.rnn(x)  # [B, lookback, hidden_dim * num_directions]
        embedding = rnn_output[:, -1, :]  # [B, hidden_dim * num_directions]
        forecast = self.forecast_head(embedding)  # [B, horizon]
        return forecast, embedding


if __name__ == "__main__":
    print("=== NDVILSTMForecaster smoke test ===")

    BATCH_SIZE = 16
    LOOKBACK = 12
    INPUT_DIM = 2  # e.g. [NDVI, EVI]
    HORIZON = 4

    # --- LSTM variant ---
    lstm_model = NDVILSTMForecaster(
        input_dim=INPUT_DIM,
        hidden_dim=64,
        num_layers=2,
        horizon=HORIZON,
        rnn_type="lstm",
    )
    print(lstm_model)

    dummy_input = torch.randn(BATCH_SIZE, LOOKBACK, INPUT_DIM)  # [16, 12, 2]
    forecast_lstm = lstm_model(dummy_input)  # [16, 4]
    print(f"LSTM forecast shape: {tuple(forecast_lstm.shape)}")
    assert forecast_lstm.shape == (BATCH_SIZE, HORIZON)

    forecast_with_emb, embedding = lstm_model.forward_with_embedding(dummy_input)
    print(f"Embedding shape: {tuple(embedding.shape)}")
    assert embedding.shape == (BATCH_SIZE, 64)
    assert forecast_with_emb.shape == (BATCH_SIZE, HORIZON)

    # --- GRU variant ---
    gru_model = NDVILSTMForecaster(
        input_dim=INPUT_DIM,
        hidden_dim=32,
        num_layers=3,
        horizon=HORIZON,
        rnn_type="gru",
        bidirectional=True,
    )
    forecast_gru = gru_model(dummy_input)  # [16, 4]
    print(f"Bidirectional GRU forecast shape: {tuple(forecast_gru.shape)}")
    assert forecast_gru.shape == (BATCH_SIZE, HORIZON)

    # --- Univariate single-layer sanity check ---
    single_feature_model = NDVILSTMForecaster(input_dim=1, hidden_dim=16, num_layers=1, horizon=1)
    univariate_input = torch.randn(BATCH_SIZE, LOOKBACK, 1)
    single_step_forecast = single_feature_model(univariate_input)
    print(f"Single-step univariate forecast shape: {tuple(single_step_forecast.shape)}")
    assert single_step_forecast.shape == (BATCH_SIZE, 1)

    num_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
    print(f"LSTM model trainable parameters: {num_params:,}")

    print("All NDVILSTMForecaster assertions passed.")
