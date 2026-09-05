"""
功能概述：将 QM9S Parquet 光谱转换为对比学习可消费的数据集，避免训练代码直接处理表格细节。
主要职责：``SpectralDataset`` 读取波数、SMILES 元数据和强度列，``__getitem__`` 生成同一光谱的
anchor/positive 视图；``create_dataloader`` 按批次、采样器和 worker 配置封装 PyTorch DataLoader。
代码库关系：依赖 ``spectraexpert.utils.augmentations.create_positive_pair``，由
``spectraexpert.train_contrastive`` 导入；数据格式为含 ``mol_id`` 的 Parquet 文件。
输入输出与副作用：构造时读取 Parquet，取样时返回两个 float Tensor 与 mol_id；归一化零方差输入
会安全地中心化而非除零，函数不修改源数据。
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Add parent to path if running as script
_parent = Path(__file__).resolve().parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from spectraexpert.utils.augmentations import create_positive_pair


class SpectralDataset(Dataset):
    """
    Dataset that loads spectra from QM9S Parquet files and constructs
    positive pairs via identity-preserving augmentations.

    Each item is (anchor_spectrum, positive_spectrum, mol_id).
    """

    def __init__(
        self,
        spectra_path: str,
        smiles_path: str,
        wavenumber_path: str,
        normalize: bool = True,
        max_samples: int = None,
    ):
        """
        Parameters
        ----------
        spectra_path : path to spectra.parquet.
        smiles_path : path to smiles.parquet.
        wavenumber_path : path to wavenumber.parquet.
        normalize : if True, standardise each spectrum to zero-mean unit-var.
        max_samples : if set, limit dataset size (for quick debugging).
        """
        super().__init__()

        # Load wavenumber axis
        wavenumber_df = pd.read_parquet(wavenumber_path)
        self.wavenumber_axis = wavenumber_df["wavenumber"].values.astype(np.float32)

        # Load SMILES metadata
        smiles_df = pd.read_parquet(smiles_path)
        self.mol_ids = smiles_df["mol_id"].values.astype(np.int64)

        # Load spectra (lazy — memory-map the Parquet for efficiency)
        # spectra.parquet has columns: mol_id, w_500, w_501, ..., w_4000
        self.spectra_df = pd.read_parquet(spectra_path)
        self.normalize = normalize

        # Extract intensity columns (all columns after mol_id)
        self.spec_cols = [c for c in self.spectra_df.columns if c != "mol_id"]
        self.n_pixels = len(self.spec_cols)
        print(f"SpectralDataset: {len(self.spectra_df)} samples, "
              f"{self.n_pixels} pixels per spectrum")
        print(f"  Wavenumber range: {self.wavenumber_axis[0]:.1f} - "
              f"{self.wavenumber_axis[-1]:.1f} cm⁻¹")

        if max_samples is not None and max_samples < len(self.spectra_df):
            self.spectra_df = self.spectra_df.iloc[:max_samples]
            self.mol_ids = self.mol_ids[:max_samples]
            print(f"  Limited to {max_samples} samples for debugging")

    def _get_spectrum(self, index: int) -> np.ndarray:
        """Retrieve a single spectrum as a 1D numpy array."""
        row = self.spectra_df.iloc[index]
        spectrum = row[self.spec_cols].values.astype(np.float32)
        return spectrum

    def _normalize(self, spectrum: np.ndarray) -> np.ndarray:
        """Standardise to zero-mean, unit-variance."""
        std = spectrum.std()
        if std > 1e-10:
            return (spectrum - spectrum.mean()) / std
        return spectrum - spectrum.mean()

    def __len__(self) -> int:
        return len(self.spectra_df)

    def __getitem__(self, index: int) -> tuple:
        """
        Returns
        -------
        anchor : torch.Tensor — shape (N,) original spectrum (lightly normalised).
        positive : torch.Tensor — shape (N,) augmented spectrum.
        mol_id : int — molecule identifier.
        """
        spectrum = self._get_spectrum(index)

        # Create positive pair
        anchor, positive = create_positive_pair(spectrum, self.wavenumber_axis)

        if self.normalize:
            anchor = self._normalize(anchor)
            positive = self._normalize(positive)

        return (
            torch.from_numpy(anchor),
            torch.from_numpy(positive),
            torch.tensor(self.mol_ids[index], dtype=torch.long),
        )


def create_dataloader(
    spectra_path: str,
    smiles_path: str,
    wavenumber_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    sampler: torch.utils.data.Sampler = None,
    num_workers: int = 4,
    normalize: bool = True,
    max_samples: int = None,
) -> DataLoader:
    """
    Convenience function to create a DataLoader for contrastive learning.

    Parameters
    ----------
    spectra_path, smiles_path, wavenumber_path : paths to Parquet files.
    batch_size : batch size (per-device in DDP mode).
    shuffle : whether to shuffle (must be False when using a sampler).
    sampler : optional torch Sampler (e.g. DistributedSampler for DDP).
    num_workers : data-loading workers.
    normalize : standardise spectra.
    max_samples : limit dataset size.

    Returns
    -------
    DataLoader yielding (anchor, positive, mol_id) batches.
    """
    dataset = SpectralDataset(
        spectra_path=spectra_path,
        smiles_path=smiles_path,
        wavenumber_path=wavenumber_path,
        normalize=normalize,
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler is not None else shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # required for NT-Xent loss (fixed batch_size)
    )
    return loader
