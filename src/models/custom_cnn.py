"""
custom_cnn.py

Baseline 2D Convolutional Neural Network for multi-spectral satellite patch
classification (e.g. EuroSAT-style land-cover classes from N-band Sentinel-2
imagery). Built from scratch (no pretrained weights) to serve as a curriculum
counterpart to the transfer-learning ResNet model.

Architecture:
    [Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d] x num_blocks
    -> AdaptiveAvgPool2d(1) -> Flatten -> Dense head (with Dropout)

Author: Principal ML Engineering Team
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn


class ConvBlock(nn.Module):
    """A single Conv -> BatchNorm -> ReLU -> MaxPool block.

    Args:
        in_channels: Number of input channels to the convolution.
        out_channels: Number of output feature maps.
        kernel_size: Convolution kernel size.
        pool_size: Max-pooling kernel size / stride (spatial downsample factor).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # "same" padding for odd kernel sizes
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_channels, height, width]
        Returns:
            [batch_size, out_channels, height // pool_size, width // pool_size]
        """
        x = self.conv(x)  # [B, out_channels, H, W]
        x = self.bn(x)  # [B, out_channels, H, W]
        x = self.act(x)  # [B, out_channels, H, W]
        x = self.pool(x)  # [B, out_channels, H/pool, W/pool]
        return x


class CustomCNN(nn.Module):
    """Baseline CNN classifier for multi-spectral satellite patches.

    Applies a stack of convolutional blocks that progressively increase
    channel depth while downsampling spatial resolution, then a global
    average pool and a small dense classification head. Designed to accept
    an arbitrary number of input channels (e.g. 3 for RGB subsets, 13 for
    full Sentinel-2 L2A).

    Args:
        in_channels: Number of input spectral bands, C.
        num_classes: Number of output land-cover classes.
        conv_channels: Output channel width for each conv block, e.g.
            [32, 64, 128, 256]. Depth of the feature extractor equals
            `len(conv_channels)`.
        kernel_size: Convolution kernel size shared across blocks.
        pool_size: Spatial downsampling factor per block.
        hidden_dim: Width of the hidden dense layer in the classification head.
        dropout_prob: Dropout probability in the classification head.

    Shape:
        - Input: `[batch_size, in_channels, height, width]`
        - Output: `[batch_size, num_classes]` (raw logits)

    Example:
        >>> model = CustomCNN(in_channels=13, num_classes=10)
        >>> x = torch.randn(8, 13, 64, 64)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([8, 10])
    """

    def __init__(
        self,
        in_channels: int = 13,
        num_classes: int = 10,
        conv_channels: Sequence[int] = (32, 64, 128, 256),
        kernel_size: int = 3,
        pool_size: int = 2,
        hidden_dim: int = 128,
        dropout_prob: float = 0.3,
    ) -> None:
        super().__init__()

        if len(conv_channels) == 0:
            raise ValueError("conv_channels must contain at least one block width.")

        blocks: List[nn.Module] = []
        previous_channels = in_channels
        for out_channels in conv_channels:
            blocks.append(
                ConvBlock(
                    in_channels=previous_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    pool_size=pool_size,
                )
            )
            previous_channels = out_channels

        self.feature_extractor = nn.Sequential(*blocks)

        # Global average pool makes the model agnostic to input spatial size,
        # so patches of varying H/W (as long as >= downsample factor) all work.
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier_head = nn.Sequential(
            nn.Flatten(),  # [B, previous_channels, 1, 1] -> [B, previous_channels]
            nn.Linear(previous_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, num_classes),
        )

        self.final_feature_channels = previous_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: FloatTensor, shape [batch_size, in_channels, height, width]

        Returns:
            FloatTensor (raw logits), shape [batch_size, num_classes]
        """
        # x: [B, C, H, W]
        features = self.feature_extractor(x)  # [B, final_channels, H', W']
        pooled = self.global_pool(features)  # [B, final_channels, 1, 1]
        logits = self.classifier_head(pooled)  # [B, num_classes]
        return logits

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the pooled feature embedding before the classification head.

        Useful for downstream tasks (clustering, similarity search, transfer
        to another head) without re-running the full forward pass.

        Args:
            x: FloatTensor, shape [batch_size, in_channels, height, width]

        Returns:
            FloatTensor, shape [batch_size, final_feature_channels]
        """
        features = self.feature_extractor(x)  # [B, final_channels, H', W']
        pooled = self.global_pool(features).flatten(1)  # [B, final_channels]
        return pooled


if __name__ == "__main__":
    print("=== CustomCNN smoke test ===")

    BATCH_SIZE = 8
    IN_CHANNELS = 13  # full Sentinel-2 L2A band stack
    NUM_CLASSES = 10
    PATCH_SIZE = 64

    model = CustomCNN(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        conv_channels=[32, 64, 128, 256],
    )
    print(model)

    dummy_input = torch.randn(BATCH_SIZE, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)  # [8, 13, 64, 64]
    logits = model(dummy_input)  # [8, 10]
    print(f"Logits shape: {tuple(logits.shape)}")
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    embedding = model.extract_embedding(dummy_input)  # [8, 256]
    print(f"Embedding shape: {tuple(embedding.shape)}")
    assert embedding.shape == (BATCH_SIZE, model.final_feature_channels)

    # Verify it also works with a different (smaller) patch size due to global pooling.
    small_input = torch.randn(BATCH_SIZE, IN_CHANNELS, 32, 32)
    small_logits = model(small_input)
    print(f"Logits shape for 32x32 input: {tuple(small_logits.shape)}")
    assert small_logits.shape == (BATCH_SIZE, NUM_CLASSES)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    print("All CustomCNN assertions passed.")
