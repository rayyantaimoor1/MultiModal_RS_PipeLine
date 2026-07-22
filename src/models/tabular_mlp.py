"""
tabular_mlp.py

A configurable Multi-Layer Perceptron for continuous-target regression on
tabular climate/weather feature vectors (e.g. predicting volumetric soil
moisture from ERA5-Land covariates).

Architecture per hidden layer:
    Linear -> BatchNorm1d -> ReLU -> Dropout

Author: Principal ML Engineering Team
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn


class SoilMoistureMLP(nn.Module):
    """MLP regressor for tabular soil-moisture (or similar) prediction.

    Stacks an arbitrary number of hidden layers, each with Batch
    Normalization, ReLU activation, and Dropout, followed by a final
    linear projection to a single continuous output.

    Args:
        input_dim: Number of input tabular features, F.
        hidden_dims: Sizes of each hidden layer, e.g. [128, 64, 32]. The
            depth of the network equals `len(hidden_dims)`.
        output_dim: Number of regression targets. Defaults to 1 (single
            scalar target, e.g. soil moisture). Set >1 for multi-target
            regression (e.g. predicting soil moisture at multiple depths).
        dropout_prob: Dropout probability applied after each hidden
            activation.
        use_batchnorm: Whether to apply BatchNorm1d after each Linear layer.
            Disable for very small batch sizes where BatchNorm statistics
            are unstable.

    Shape:
        - Input: `[batch_size, input_dim]`
        - Output: `[batch_size, output_dim]`

    Example:
        >>> model = SoilMoistureMLP(input_dim=7, hidden_dims=[128, 64], output_dim=1)
        >>> x = torch.randn(32, 7)
        >>> y = model(x)
        >>> y.shape
        torch.Size([32, 1])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64, 32),
        output_dim: int = 1,
        dropout_prob: float = 0.2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size.")

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers: List[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_prob))
            previous_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_head = nn.Linear(previous_dim, output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Applies Kaiming (He) initialization, appropriate for ReLU networks."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: FloatTensor, shape [batch_size, input_dim]

        Returns:
            FloatTensor, shape [batch_size, output_dim]
        """
        # x: [batch_size, input_dim]
        hidden = self.hidden_layers(x)  # [batch_size, hidden_dims[-1]]
        output = self.output_head(hidden)  # [batch_size, output_dim]
        return output


if __name__ == "__main__":
    print("=== SoilMoistureMLP smoke test ===")

    BATCH_SIZE = 32
    INPUT_DIM = 7  # e.g. t2m, d2m, tp, ssrd, sp, u10, v10

    model = SoilMoistureMLP(
        input_dim=INPUT_DIM,
        hidden_dims=[128, 64, 32],
        output_dim=1,
        dropout_prob=0.2,
        use_batchnorm=True,
    )
    print(model)

    dummy_input = torch.randn(BATCH_SIZE, INPUT_DIM)  # [32, 7]
    model.train()
    output_train = model(dummy_input)  # [32, 1]
    print(f"Train-mode output shape: {tuple(output_train.shape)}")
    assert output_train.shape == (BATCH_SIZE, 1)

    model.eval()
    with torch.no_grad():
        output_eval = model(dummy_input)  # [32, 1]
    print(f"Eval-mode output shape: {tuple(output_eval.shape)}")
    assert output_eval.shape == (BATCH_SIZE, 1)

    # Multi-target regression variant (e.g. soil moisture at 3 depths).
    multi_target_model = SoilMoistureMLP(input_dim=INPUT_DIM, hidden_dims=[64, 32], output_dim=3)
    multi_output = multi_target_model(dummy_input)  # [32, 3]
    print(f"Multi-target output shape: {tuple(multi_output.shape)}")
    assert multi_output.shape == (BATCH_SIZE, 3)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    print("All SoilMoistureMLP assertions passed.")
