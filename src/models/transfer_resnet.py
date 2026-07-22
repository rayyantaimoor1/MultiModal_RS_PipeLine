"""
transfer_resnet.py

Transfer-learning model that adapts a torchvision ResNet-50 (pretrained on
ImageNet RGB imagery) to accept N-channel multi-spectral satellite input
(e.g. all 13 Sentinel-2 L2A bands) instead of 3-channel RGB.

Key adaptation steps:
    1. Replace `conv1` (originally Conv2d(3, 64, kernel_size=7, stride=2))
       with a new Conv2d(N, 64, ...) layer. The new layer's weights are
       initialized by tiling/averaging the pretrained RGB kernels across
       the extra spectral channels, which preserves useful low-level
       edge/texture filters instead of starting from random weights.
    2. Replace the final `fc` classification head with a new head sized
       to the target number of classes.
    3. Provide layer-freezing utilities for staged fine-tuning (freeze
       backbone -> train head only -> unfreeze last blocks -> fine-tune).

Author: Principal ML Engineering Team
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn
from torchvision import models


class MultiSpectralResNet(nn.Module):
    """ResNet-50 backbone adapted for N-channel multi-spectral classification.

    Args:
        num_channels: Number of input spectral bands (e.g. 13 for full
            Sentinel-2 L2A, 4 for a RGB+NIR subset).
        num_classes: Number of output classes for the new classification head.
        pretrained: If True, loads ImageNet-pretrained weights for all layers
            that are not shape-incompatible (i.e. everything except `conv1`,
            which is rebuilt to match `num_channels`).
        freeze_backbone: If True, all backbone parameters (everything except
            the new `fc` head) start frozen (`requires_grad=False`). Useful
            for a first-stage "train head only" fine-tuning phase.
        dropout_prob: Dropout probability applied immediately before the
            final linear classification layer.

    Shape:
        - Input: `[batch_size, num_channels, height, width]` (height/width
          should be >= 32; ResNet-50 was designed around 224x224 but is
          fully convolutional up to the global average pool, so other
          sizes work via adaptive pooling).
        - Output: `[batch_size, num_classes]` (raw logits)

    Example:
        >>> model = MultiSpectralResNet(num_channels=13, num_classes=10, pretrained=True)
        >>> x = torch.randn(4, 13, 224, 224)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([4, 10])
    """

    def __init__(
        self,
        num_channels: int = 13,
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_prob: float = 0.3,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        original_conv1 = backbone.conv1  # Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv1 = self._build_adapted_conv1(original_conv1, num_channels)
        backbone.conv1 = new_conv1

        # Replace classification head; capture the backbone feature width first.
        in_features = backbone.fc.in_features  # 2048 for ResNet-50
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features, num_classes),
        )

        self.backbone = backbone
        self.feature_dim = in_features

        if freeze_backbone:
            self.freeze_backbone()

    @staticmethod
    def _build_adapted_conv1(original_conv1: nn.Conv2d, num_channels: int) -> nn.Conv2d:
        """Builds a new first conv layer sized for `num_channels`, seeded from RGB weights.

        Rather than random-initializing the new `conv1`, this reuses the
        pretrained RGB filters (which encode useful edge/texture detectors)
        by averaging them across the RGB dimension and replicating that
        average across every new spectral channel. This gives fine-tuning a
        much better starting point than random init, especially with
        limited satellite-labeled data.

        Args:
            original_conv1: The pretrained Conv2d(3, 64, ...) layer.
            num_channels: Desired number of input channels, N.

        Returns:
            A new Conv2d(N, 64, ...) layer with seeded weights.
        """
        new_conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=(original_conv1.bias is not None),
        )

        with torch.no_grad():
            # original weight: [out_channels, 3, kH, kW]
            original_weight = original_conv1.weight.data
            # Average across the RGB input-channel dim -> [out_channels, 1, kH, kW]
            mean_weight = original_weight.mean(dim=1, keepdim=True)
            # Tile the averaged filter across all N new input channels.
            tiled_weight = mean_weight.repeat(1, num_channels, 1, 1)  # [out_channels, N, kH, kW]
            new_conv1.weight.data.copy_(tiled_weight)

            if original_conv1.bias is not None and new_conv1.bias is not None:
                new_conv1.bias.data.copy_(original_conv1.bias.data)

        return new_conv1

    def freeze_backbone(self) -> None:
        """Freezes every parameter except the final `fc` classification head.

        Intended for a first fine-tuning stage: train only the new head on
        top of frozen pretrained (or seeded) features.
        """
        for name, param in self.backbone.named_parameters():
            param.requires_grad = name.startswith("fc.")

    def unfreeze_all(self) -> None:
        """Unfreezes every parameter in the backbone for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_n_blocks(self, n: int = 1) -> None:
        """Unfreezes the last `n` ResNet stages (layer4, layer3, ...) plus the head.

        Enables a staged fine-tuning schedule: start with only the head
        trainable, then progressively unfreeze deeper layers as validation
        loss plateaus, which tends to stabilize transfer learning on small
        labeled satellite datasets.

        Args:
            n: Number of trailing residual stages to unfreeze (1 to 4).
                `n=1` unfreezes `layer4` only; `n=4` unfreezes the entire
                backbone including `conv1`/`bn1`.
        """
        stage_names = ["layer4", "layer3", "layer2", "layer1"]
        n = max(0, min(n, len(stage_names)))
        stages_to_unfreeze = set(stage_names[:n])

        for name, param in self.backbone.named_parameters():
            if name.startswith("fc."):
                param.requires_grad = True
            elif any(name.startswith(stage) for stage in stages_to_unfreeze):
                param.requires_grad = True
            elif n >= len(stage_names) and (name.startswith("conv1") or name.startswith("bn1")):
                param.requires_grad = True

    def get_trainable_parameter_groups(self) -> List[torch.nn.Parameter]:
        """Returns the list of currently trainable parameters (requires_grad=True)."""
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the adapted ResNet-50.

        Args:
            x: FloatTensor, shape [batch_size, num_channels, height, width]

        Returns:
            FloatTensor (raw logits), shape [batch_size, num_classes]
        """
        # x: [B, num_channels, H, W]
        logits = self.backbone(x)  # [B, num_classes]
        return logits

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the 2048-d global-average-pooled feature embedding (pre-fc).

        Args:
            x: FloatTensor, shape [batch_size, num_channels, height, width]

        Returns:
            FloatTensor, shape [batch_size, feature_dim] (feature_dim=2048 for ResNet-50)
        """
        # Manually replay the backbone forward pass up to (but excluding) `fc`.
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        x = b.avgpool(x)  # [B, 2048, 1, 1]
        embedding = torch.flatten(x, 1)  # [B, 2048]
        return embedding


if __name__ == "__main__":
    print("=== MultiSpectralResNet smoke test ===")

    BATCH_SIZE = 4
    NUM_CHANNELS = 13  # full Sentinel-2 L2A stack
    NUM_CLASSES = 10
    IMAGE_SIZE = 128  # smaller than the canonical 224 to keep the smoke test fast

    model = MultiSpectralResNet(
        num_channels=NUM_CHANNELS,
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_backbone=True,
    )

    print(f"conv1: {model.backbone.conv1}")
    print(f"fc head: {model.backbone.fc}")

    dummy_input = torch.randn(BATCH_SIZE, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)  # [4, 13, 128, 128]

    logits = model(dummy_input)  # [4, 10]
    print(f"Logits shape: {tuple(logits.shape)}")
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    embedding = model.extract_embedding(dummy_input)  # [4, 2048]
    print(f"Embedding shape: {tuple(embedding.shape)}")
    assert embedding.shape == (BATCH_SIZE, model.feature_dim)

    trainable_frozen = sum(p.numel() for p in model.get_trainable_parameter_groups())
    print(f"Trainable params (backbone frozen, head only): {trainable_frozen:,}")

    model.unfreeze_last_n_blocks(n=1)
    trainable_stage1 = sum(p.numel() for p in model.get_trainable_parameter_groups())
    print(f"Trainable params (layer4 + head unfrozen): {trainable_stage1:,}")
    assert trainable_stage1 > trainable_frozen

    model.unfreeze_all()
    trainable_all = sum(p.numel() for p in model.get_trainable_parameter_groups())
    print(f"Trainable params (fully unfrozen): {trainable_all:,}")
    assert trainable_all > trainable_stage1

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("All MultiSpectralResNet assertions passed.")
