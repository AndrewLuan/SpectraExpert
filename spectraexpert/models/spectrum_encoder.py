"""
Spectrum Transformer Encoder — a 1D ViT-style transformer that processes
Raman spectra as patch sequences.

Architecture (inspired by MatExpert's dual-T5 SimCLR, but adapted for
1D spectral data):

  1. Raw spectrum (3501 points) → overlapping 1D patches
  2. Linear projection → patch embeddings
  3. Learnable positional encoding
  4. Transformer encoder layers (with pre-norm, GeLU)
  5. Global average pooling → projection head → L2-normalised embedding

The projection head follows SimCLR's design:
    Linear(d_model, d_model) → ReLU → Linear(d_model, projection_dim)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed1D(nn.Module):
    """
    1D patch embedding for spectral data.

    Splits a 1D signal into (possibly overlapping) patches and projects
    each patch to a D-dimensional embedding.
    """

    def __init__(self, signal_length: int, patch_size: int = 16,
                 stride: int = 8, embed_dim: int = 768):
        """
        Parameters
        ----------
        signal_length : length of the input 1D spectrum.
        patch_size : size of each patch (in pixels / wavenumber steps).
        stride : stride between consecutive patches (<= patch_size for overlap).
        embed_dim : output embedding dimension per patch.
        """
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.signal_length = signal_length

        # Number of patches
        self.num_patches = (signal_length - patch_size) // stride + 1

        # Linear projection: patch → embedding
        self.proj = nn.Linear(patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L) tensor — raw spectra.

        Returns
        -------
        patches : (B, num_patches, embed_dim) tensor.
        """
        B, L = x.shape
        assert L == self.signal_length, \
            f"Expected signal length {self.signal_length}, got {L}"

        # Unfold into overlapping patches
        patches = x.unfold(1, self.patch_size, self.stride)  # (B, num_patches, patch_size)
        patches = self.proj(patches)  # (B, num_patches, embed_dim)
        return patches


class SpectrumTransformer(nn.Module):
    """
    Transformer encoder for 1D Raman spectra.

    Designed as a drop-in replacement for MatExpert's T5Encoder in the
    SimCLR framework, but operating directly on raw spectral data rather
    than on text token sequences.
    """

    def __init__(
        self,
        signal_length: int = 3501,
        patch_size: int = 16,
        stride: int = 8,
        embed_dim: int = 768,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        projection_dim: int = 256,
    ):
        """
        Parameters
        ----------
        signal_length : length of input 1D spectrum.
        patch_size : pixels per patch.
        stride : patch stride.
        embed_dim : transformer embedding dimension.
        depth : number of transformer encoder layers.
        num_heads : attention heads per layer.
        mlp_ratio : hidden dim ratio in FFN.
        dropout : dropout rate.
        projection_dim : output embedding dimension (after projection head).
        """
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbed1D(
            signal_length=signal_length,
            patch_size=patch_size,
            stride=stride,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder layers (pre-norm architecture)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # Projection head (SimCLR-style)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, projection_dim, bias=False),
        )

        # Initialise weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L) tensor — raw spectra.

        Returns
        -------
        z : (B, projection_dim) tensor — L2-normalised embeddings.
        """
        # Patch embed + positional encoding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer encoder
        x = self.encoder(x)  # (B, num_patches, embed_dim)
        x = self.norm(x)

        # Global average pooling over patches
        x = x.mean(dim=1)  # (B, embed_dim)

        # Projection head
        z = self.projection(x)  # (B, projection_dim)

        # L2 normalise
        z = F.normalize(z, dim=1)

        return z


class SpectrumSimCLR(nn.Module):
    """
    SimCLR framework for spectral data.

    Unlike MatExpert's dual-encoder SimCLR (which aligns property↔structure
    text descriptions), this uses a single SpectrumTransformer encoder
    and produces embeddings for both views (anchor & positive) by sharing
    weights — standard SimCLR formulation.
    """

    def __init__(self, encoder: SpectrumTransformer):
        super().__init__()
        self.encoder = encoder

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> tuple:
        """
        Parameters
        ----------
        x_i : (B, L) — anchor batch.
        x_j : (B, L) — positive batch.

        Returns
        -------
        (z_i, z_j) — both (B, projection_dim), L2-normalised.
        """
        z_i = self.encoder(x_i)
        z_j = self.encoder(x_j)
        return z_i, z_j
