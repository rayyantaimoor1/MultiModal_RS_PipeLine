"""
spatial_vit.py

A minimal Vision Transformer (ViT), built from scratch, for classifying
multi-spectral satellite raster patches. Implements the core ViT recipe
(Dosovitskiy et al., 2020) directly on top of PyTorch's built-in
`nn.TransformerEncoder`, rather than importing a pretrained ViT — useful
as a curriculum step showing exactly how patch embedding, the [CLS] token,
and positional embeddings compose into a working transformer classifier.

Pipeline:
    raster [B, C, H, W]
      -> PatchEmbedding (Conv2d "patchify" + linear projection) -> [B, N, D]
      -> prepend learnable [CLS] token                          -> [B, N+1, D]
      -> add learnable positional embeddings                    -> [B, N+1, D]
      -> nn.TransformerEncoder (stack of self-attention blocks)  -> [B, N+1, D]
      -> take [CLS] token output -> classification head          -> [B, num_classes]

Author: Principal ML Engineering Team
"""

from __future__ import annotations

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Splits an input raster into non-overlapping patches and linearly projects them.

    Implemented as a single strided Conv2d, which is mathematically
    equivalent to "reshape into patches + flatten + linear projection" but
    is more efficient and numerically convenient in PyTorch.

    Args:
        in_channels: Number of input spectral bands, C.
        patch_size: Side length of each square patch, P (patches are
            P x P pixels). `image_size` must be divisible by `patch_size`.
        embed_dim: Output embedding dimension, D, for each patch token.
        image_size: Expected input spatial size (assumes square images,
            H == W == image_size). Used to precompute `num_patches`.

    Shape:
        - Input: `[batch_size, in_channels, image_size, image_size]`
        - Output: `[batch_size, num_patches, embed_dim]`
    """

    def __init__(
        self,
        in_channels: int = 13,
        patch_size: int = 8,
        embed_dim: int = 128,
        image_size: int = 64,
    ) -> None:
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size ({patch_size})."
            )

        self.patch_size = patch_size
        self.image_size = image_size
        self.grid_size = image_size // patch_size  # patches per side
        self.num_patches = self.grid_size * self.grid_size

        # A Conv2d with kernel_size == stride == patch_size extracts each
        # P x P patch and projects it to `embed_dim` in a single operation:
        # equivalent to flattening each patch and applying a Linear layer.
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: FloatTensor, shape [batch_size, in_channels, image_size, image_size]

        Returns:
            FloatTensor, shape [batch_size, num_patches, embed_dim]
        """
        # x: [B, C, H, W]
        projected = self.projection(x)  # [B, embed_dim, grid_size, grid_size]
        flattened = projected.flatten(2)  # [B, embed_dim, num_patches]
        patch_tokens = flattened.transpose(1, 2)  # [B, num_patches, embed_dim]
        return patch_tokens


class MinimalGeospatialViT(nn.Module):
    """A from-scratch Vision Transformer for multi-spectral satellite patch classification.

    Combines a custom `PatchEmbedding` layer with a learnable `[CLS]`
    token, learnable positional embeddings, and PyTorch's
    `nn.TransformerEncoder` for self-attention over the patch-token
    sequence. The final `[CLS]` token representation is passed through a
    small MLP head to produce class logits.

    Args:
        in_channels: Number of input spectral bands, C (e.g. 13 for full
            Sentinel-2 L2A).
        image_size: Expected input spatial size (square images assumed).
        patch_size: Side length of each square patch.
        embed_dim: Token embedding dimension, D.
        num_heads: Number of self-attention heads per encoder layer.
        num_encoder_layers: Number of stacked `TransformerEncoderLayer`s.
        mlp_ratio: Expansion ratio for the feed-forward hidden dimension
            inside each encoder layer (hidden_dim = embed_dim * mlp_ratio).
        num_classes: Number of output land-cover classes.
        dropout_prob: Dropout probability used in the transformer encoder
            layers and the final classification head.

    Shape:
        - Input: `[batch_size, in_channels, image_size, image_size]`
        - Output: `[batch_size, num_classes]` (raw logits)

    Example:
        >>> model = MinimalGeospatialViT(in_channels=13, image_size=64, patch_size=8, num_classes=10)
        >>> x = torch.randn(4, 13, 64, 64)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([4, 10])
    """

    def __init__(
        self,
        in_channels: int = 13,
        image_size: int = 64,
        patch_size: int = 8,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 6,
        mlp_ratio: int = 4,
        num_classes: int = 10,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            image_size=image_size,
        )
        num_patches = self.patch_embedding.num_patches

        # Learnable [CLS] token, prepended to the patch-token sequence.
        # Shape [1, 1, embed_dim] so it can be expanded/broadcast per batch.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable positional embeddings for [CLS] + all patch tokens.
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(p=dropout_prob)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN, more stable training for ViTs
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.final_norm = nn.LayerNorm(embed_dim)
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(embed_dim, num_classes),
        )

        self.embed_dim = embed_dim
        self.num_patches = num_patches

        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes [CLS] token and positional embeddings with small random values."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: FloatTensor, shape [batch_size, in_channels, image_size, image_size]

        Returns:
            FloatTensor (raw logits), shape [batch_size, num_classes]
        """
        batch_size = x.shape[0]

        # x: [B, C, H, W]
        patch_tokens = self.patch_embedding(x)  # [B, num_patches, embed_dim]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, embed_dim]
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)  # [B, num_patches + 1, embed_dim]

        tokens = tokens + self.positional_embedding  # broadcast add: [B, N+1, D]
        tokens = self.dropout(tokens)

        encoded = self.transformer_encoder(tokens)  # [B, num_patches + 1, embed_dim]
        encoded = self.final_norm(encoded)  # [B, num_patches + 1, embed_dim]

        cls_output = encoded[:, 0, :]  # [B, embed_dim] -- the [CLS] token's final representation
        logits = self.classification_head(cls_output)  # [B, num_classes]
        return logits

    def extract_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the encoder's per-patch token representations (excluding [CLS]).

        Useful for dense/segmentation-style downstream tasks or attention
        visualization over the spatial grid.

        Args:
            x: FloatTensor, shape [batch_size, in_channels, image_size, image_size]

        Returns:
            FloatTensor, shape [batch_size, num_patches, embed_dim]
        """
        batch_size = x.shape[0]
        patch_tokens = self.patch_embedding(x)  # [B, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, embed_dim]
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1) + self.positional_embedding
        encoded = self.transformer_encoder(self.dropout(tokens))
        encoded = self.final_norm(encoded)
        return encoded[:, 1:, :]  # [B, num_patches, embed_dim] -- drop [CLS]


if __name__ == "__main__":
    print("=== MinimalGeospatialViT smoke test ===")

    BATCH_SIZE = 4
    IN_CHANNELS = 13  # full Sentinel-2 L2A stack
    IMAGE_SIZE = 64
    PATCH_SIZE = 8
    NUM_CLASSES = 10

    model = MinimalGeospatialViT(
        in_channels=IN_CHANNELS,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=128,
        num_heads=4,
        num_encoder_layers=6,
        num_classes=NUM_CLASSES,
    )
    print(f"num_patches: {model.num_patches} (grid {IMAGE_SIZE // PATCH_SIZE}x{IMAGE_SIZE // PATCH_SIZE})")

    dummy_input = torch.randn(BATCH_SIZE, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)  # [4, 13, 64, 64]

    # --- Standalone PatchEmbedding check ---
    patch_embed = PatchEmbedding(in_channels=IN_CHANNELS, patch_size=PATCH_SIZE, embed_dim=128, image_size=IMAGE_SIZE)
    patch_tokens = patch_embed(dummy_input)  # [4, 64, 128]
    print(f"PatchEmbedding output shape: {tuple(patch_tokens.shape)}")
    assert patch_tokens.shape == (BATCH_SIZE, model.num_patches, 128)

    # --- Full ViT forward pass ---
    logits = model(dummy_input)  # [4, 10]
    print(f"ViT logits shape: {tuple(logits.shape)}")
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    patch_embeddings = model.extract_patch_embeddings(dummy_input)  # [4, 64, 128]
    print(f"Per-patch embedding shape: {tuple(patch_embeddings.shape)}")
    assert patch_embeddings.shape == (BATCH_SIZE, model.num_patches, model.embed_dim)

    # --- Different image/patch configuration sanity check ---
    small_patch_model = MinimalGeospatialViT(
        in_channels=4, image_size=32, patch_size=4, embed_dim=64, num_heads=2, num_encoder_layers=2, num_classes=5
    )
    small_input = torch.randn(2, 4, 32, 32)
    small_logits = small_patch_model(small_input)
    print(f"Alt-config logits shape: {tuple(small_logits.shape)}")
    assert small_logits.shape == (2, 5)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    print("All MinimalGeospatialViT assertions passed.")
