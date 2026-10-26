"""功能概述：SpecGNN NMR 迁移实验的数据契约与显式氢图特征，不读取标签生成图。
主要职责：identity 统一结构并按去立体连接图分组；graph 保留用户提供的99/24维
原子/键特征，修复负one-hot索引与无边分子；quantiles 将峰位置转为128个分位数；
condition 编码实验条件；physical_features 编码原子计数、交换H、酰胺及已认证分组计数。
关系：prepare.py/runner.py/model.py 共用此模块。图在CPU构建，训练批次才传GPU。
输入输出：无网络；不改SMILES来源或原始谱。物理计数来自结构算法，不作为真值准确率。
无强度的峰表按峰等权；积分峰表保留积分；峰高不得冒充积分。屏蔽常数转换由prepare负责。
"""
import hashlib
import re
import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog('rdApp.*')
Q = 128
SCALES = {'1H': 10., '13C': 200.}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def identity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('invalid_smiles')
    mol = Chem.RemoveHs(mol)
    if '.' in Chem.MolToSmiles(mol):
        raise ValueError('multicomponent_structure')
    iso = Chem.MolToSmiles(mol, isomericSmiles=True)
    plain = Chem.MolToSmiles(mol, isomericSmiles=False)
    group = digest(plain)
    partition = int(group[:8], 16) % 100
    return iso, group, 'train' if partition < 80 else 'val' if partition < 90 else 'test'


def hot(value, size):
    if not 0 <= value < size:
        raise ValueError(f'unsupported_feature:{value}/{size}')
    return [int(i == value) for i in range(size)]


def graph(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if not 1 <= mol.GetNumAtoms() <= 192:
        raise ValueError('atom_count_limit')
    nodes, edges, bonds = [], [], []
    for a in mol.GetAtoms():
        nodes.append(hot(a.GetAtomicNum()-1, 54) + hot(int(a.GetChiralTag()), 4)
            + hot(a.GetTotalDegree()-1, 6) + hot(a.GetFormalCharge()+3, 7)
            + hot(int(a.GetHybridization()), 8) + hot(int(a.GetIsAromatic()), 2)
            + hot(a.GetTotalValence()-1, 8) + hot(int(a.IsInRing()), 2)
            + [int(a.IsInRingSize(i)) for i in range(3, 11)])
    for b in mol.GetBonds():
        bt = b.GetBondType()
        f = [0, int(bt == Chem.BondType.SINGLE), int(bt == Chem.BondType.DOUBLE),
             int(bt == Chem.BondType.TRIPLE), int(bt == Chem.BondType.AROMATIC)]
        f += [int(b.IsInRingSize(i)) for i in range(3, 11)]
        f += hot(int(b.IsInRing()), 2) + hot(int(b.GetIsConjugated()), 2) + hot(int(b.GetStereo()), 7)
        for i, j in [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), (b.GetEndAtomIdx(), b.GetBeginAtomIdx())]:
            edges.append((i, j)); bonds.append(f)
    return Data(x=torch.tensor(nodes, dtype=torch.uint8),
                edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous(),
                edge_attr=torch.tensor(bonds, dtype=torch.uint8).reshape(-1, 24))


def quantiles(xs, weights=None):
    x = np.asarray(xs, dtype=np.float64)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=np.float64)
    if not len(x) or x.shape != w.shape or not np.isfinite(x).all() or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError('invalid_spectrum')
    order = np.argsort(x); x, w = x[order], w[order]
    indices = np.searchsorted(w.cumsum()/w.sum(), (np.arange(Q)+.5)/Q)
    return x[indices].astype(np.float32)


def condition(record):
    c = record.get('experimental_conditions') or record.get('conditions') or {}
    solvent = str(record.get('solvent') or c.get('solvent') or '').lower()
    frequency = str(record.get('frequency') or c.get('frequency') or '')
    match = re.search(r'[0-9]+(?:\.[0-9]+)?', frequency)
    return [float(match.group())/600 if match else 0.,
            float('water' in solvent or 'd2o' in solvent), float('cdcl3' in solvent), float('dmso' in solvent)]


AMIDE = Chem.MolFromSmarts('[NX3][CX3](=[OX1])')


def physical_features(smiles, certified=None):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    h = sum(a.GetAtomicNum() == 1 for a in mol.GetAtoms())
    c = sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms())
    exchange = sum(a.GetAtomicNum() == 1 and a.GetDegree() == 1 and a.GetNeighbors()[0].GetAtomicNum() in (7, 8, 16) for a in mol.GetAtoms())
    aromatic = sum(a.GetIsAromatic() for a in mol.GetAtoms())
    counts = certified or {}
    return np.array([h/40, c/30, exchange/8, len(mol.GetSubstructMatches(AMIDE))/4,
                     aromatic/30, counts.get('1H', 0)/30, counts.get('13C', 0)/30,
                     float(bool(counts))], dtype=np.float32)
