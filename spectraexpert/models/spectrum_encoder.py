"""
功能概述：把一维 Raman/IR 光谱编码成 Transformer pooled 特征及单位范数检索向量。
主要职责：PatchEmbed1D 将 (B,L) 光谱切成重叠 patch；SpectrumTransformer.encode_backbone
负责位置编码、Transformer 和均值池化，project 负责投影及 L2 归一化，forward 串联二者；
SpectrumSimCLR 共享同一 encoder 处理两个增强视图；_init_weights 初始化线性层和 LayerNorm。
代码库关系：legacy SpectralEncoder 与 train_contrastive 继续消费原 forward/state_dict；
StructureAwareRetriever 读取 backbone 构建辅助头，并用公开的构造标量保存 checkpoint 配置。
输入输出与副作用：输入 (B,signal_length) 浮点 tensor，返回 (B,embed_dim) 特征或
(B,projection_dim) embedding；训练态 dropout 随机，模块参数键保持兼容，不读取结构标签或文件。
"""
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

        self.signal_length = signal_length
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.projection_dim = projection_dim

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

    def encode_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """返回 projection 前的 (B, embed_dim) 光谱 pooled feature。"""
        # Patch embed + positional encoding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer encoder
        x = self.encoder(x)  # (B, num_patches, embed_dim)
        x = self.norm(x)

        # Global average pooling over patches
        return x.mean(dim=1)  # (B, embed_dim)

    def project(self, features: torch.Tensor) -> torch.Tensor:
        """投影到 (B, projection_dim) 并做安全 L2 归一化。"""
        return F.normalize(self.projection(features), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 (B, L) 光谱编码为 L2-normalized embedding，保持 legacy 语义。"""
        return self.project(self.encode_backbone(x))


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
