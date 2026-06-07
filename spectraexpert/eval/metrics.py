"""
Graph-based metrics for molecular structure comparison.

Primarily uses Graph Edit Distance (GED) via networkx, with Tanimoto
similarity on Morgan fingerprints as a supplementary metric.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity


# ---------------------------------------------------------------------------
# SMILES → NetworkX graph (heavy-atom only)
# ---------------------------------------------------------------------------

def smiles_to_nx(
    smiles: str,
    include_h: bool = False,
) -> Optional[nx.Graph]:
    """Convert a SMILES string to a NetworkX graph (heavy atoms only).

    Parameters
    ----------
    smiles : str
        Valid SMILES string.
    include_h : bool
        If True, include implicit hydrogens as explicit nodes.  Drastically
        increases GED computation cost; default False.

    Returns
    -------
    nx.Graph or None
        Nodes have attribute ``atomic_num`` (int).
        Edges have attribute ``bond_type`` (int, from RDKit bond type enum).
        Returns None if the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if not include_h:
        mol = Chem.RemoveHs(mol)

    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum())

    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type=int(bond.GetBondTypeAsDouble()),
        )

    return G


# ---------------------------------------------------------------------------
# Node / edge matchers for networkx graph edit distance
# ---------------------------------------------------------------------------

def _node_match(a: dict, b: dict) -> bool:
    """Match nodes by atomic number (element identity)."""
    return a.get("atomic_num") == b.get("atomic_num")


def _edge_match(a: dict, b: dict) -> bool:
    """Match edges by bond type."""
    return a.get("bond_type") == b.get("bond_type")


# ---------------------------------------------------------------------------
# Graph Edit Distance
# ---------------------------------------------------------------------------

def graph_edit_distance(
    smiles_a: str,
    smiles_b: str,
    timeout: float = 30.0,
    include_h: bool = False,
) -> Optional[float]:
    """Compute Graph Edit Distance between two molecules.

    Uses ``nx.optimize_graph_edit_distance`` with exact search (A\*-Beam).
    For molecules with few heavy atoms (< ~20) this is fast.  Larger
    molecules may time out — use ``timeout`` to bound the search.

    Parameters
    ----------
    smiles_a, smiles_b : str
        SMILES strings of the two molecules.
    timeout : float
        Maximum wall-clock seconds for the GED search.  If the search
        exceeds this, returns None (and prints a warning).
    include_h : bool
        Whether to include implicit hydrogens as nodes.

    Returns
    -------
    float or None
        The edit distance, or None if either SMILES is invalid or the
        search timed out.
    """
    Ga = smiles_to_nx(smiles_a, include_h=include_h)
    Gb = smiles_to_nx(smiles_b, include_h=include_h)
    if Ga is None or Gb is None:
        return None

    import signal

    class TimeoutError(Exception):
        pass

    def _handler(signum, frame):
        raise TimeoutError("GED search timed out")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(timeout) + 1)

    try:
        for dist in nx.optimize_graph_edit_distance(
            Ga,
            Gb,
            node_match=_node_match,
            edge_match=_edge_match,
        ):
            result = float(dist)
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            return result
    except TimeoutError:
        signal.signal(signal.SIGALRM, old_handler)
        return None
    except Exception:
        signal.signal(signal.SIGALRM, old_handler)
        return None

    signal.signal(signal.SIGALRM, old_handler)
    return None


def batch_graph_edit_distance(
    ground_truth: str,
    candidates: list[str],
    timeout: float = 10.0,
) -> list[Optional[float]]:
    """Compute GED between a ground-truth SMILES and a list of candidates.

    The ground-truth graph is constructed once and reused for all
    comparisons to avoid redundant parsing.

    Parameters
    ----------
    ground_truth : str
        Ground-truth SMILES.
    candidates : list of str
        Candidate SMILES strings.
    timeout : float
        Per-comparison timeout in seconds.

    Returns
    -------
    list of float or None
        GED for each candidate (None if timeout or parse error).
    """
    G_gt = smiles_to_nx(ground_truth)
    if G_gt is None:
        return [None] * len(candidates)

    import signal

    class TimeoutError(Exception):
        pass

    def _handler(signum, frame):
        raise TimeoutError

    old_handler = signal.signal(signal.SIGALRM, _handler)

    results = []
    for c_smi in candidates:
        G_c = smiles_to_nx(c_smi)
        if G_c is None:
            results.append(None)
            continue

        signal.alarm(int(timeout) + 1)
        try:
            for dist in nx.optimize_graph_edit_distance(
                G_gt, G_c, node_match=_node_match, edge_match=_edge_match,
            ):
                results.append(float(dist))
                break
            else:
                results.append(None)
            signal.alarm(0)
        except TimeoutError:
            results.append(None)
        except Exception:
            results.append(None)
        finally:
            signal.alarm(0)

    signal.signal(signal.SIGALRM, old_handler)
    return results


# ---------------------------------------------------------------------------
# Tanimoto similarity (Morgan fingerprints)
# ---------------------------------------------------------------------------

def tanimoto_similarity(smiles_a: str, smiles_b: str) -> Optional[float]:
    """Compute Tanimoto similarity on Morgan (ECFP-like) fingerprints.

    Parameters
    ----------
    smiles_a, smiles_b : str
        SMILES strings.

    Returns
    -------
    float (0–1) or None if either SMILES cannot be parsed.
    """
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return None

    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, 2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, 2048)
    return float(TanimotoSimilarity(fp_a, fp_b))


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    ground_truth: str,
    candidate_smiles: list[str],
    candidate_similarities: list[float],
) -> dict:
    """Compute retrieval metrics for a single query.

    Parameters
    ----------
    ground_truth : str
        Ground-truth SMILES.
    candidate_smiles : list of str
        Candidate SMILES ranked by similarity (descending).
    candidate_similarities : list of float
        Cosine similarities corresponding to candidates.

    Returns
    -------
    dict with keys:
        top1_smiles, top1_sim, top1_ged,
        min_ged_top5, min_ged_top10, min_ged_all,
        rank_of_min_ged, best_ged, best_ged_smiles,
        top1_tanimoto, tanimoto_top5, tanimoto_top10,
        num_valid_candidates.
    """
    num_candidates = len(candidate_smiles)
    if num_candidates == 0:
        return {
            "top1_smiles": "",
            "top1_sim": 0.0,
            "top1_ged": None,
            "min_ged_top5": None,
            "min_ged_top10": None,
            "min_ged_all": None,
            "rank_of_min_ged": -1,
            "best_ged": None,
            "best_ged_smiles": "",
            "top1_tanimoto": None,
            "tanimoto_top5": None,
            "tanimoto_top10": None,
            "num_valid_candidates": 0,
        }

    # --- GED for top-1 ---
    top1_ged = graph_edit_distance(ground_truth, candidate_smiles[0], timeout=10.0)

    # --- GED for all candidates (batch) ---
    all_ged = batch_graph_edit_distance(ground_truth, candidate_smiles, timeout=10.0)
    valid_ged = [g for g in all_ged if g is not None]

    # --- Tanimoto for top-1, top-5, top-10 ---
    top1_tani = tanimoto_similarity(ground_truth, candidate_smiles[0])
    tani_top5 = [
        tanimoto_similarity(ground_truth, c)
        for c in candidate_smiles[:5]
    ]
    tani_top10 = [
        tanimoto_similarity(ground_truth, c)
        for c in candidate_smiles[:10]
    ]

    min_ged_top5 = min(valid_ged[:5]) if len(valid_ged[:5]) > 0 else None
    min_ged_top10 = min(valid_ged[:10]) if len(valid_ged[:10]) > 0 else None
    min_ged_all = min(valid_ged) if valid_ged else None

    # Rank of the candidate with minimum GED
    if valid_ged and min_ged_all is not None:
        # Find first occurrence of min GED
        for rank, g in enumerate(valid_ged):
            if g == min_ged_all:
                rank_of_min_ged = rank + 1  # 1-based
                best_ged_smiles = candidate_smiles[rank]
                break
        else:
            rank_of_min_ged = -1
            best_ged_smiles = ""
    else:
        rank_of_min_ged = -1
        best_ged_smiles = ""

    return {
        "top1_smiles": candidate_smiles[0],
        "top1_sim": float(candidate_similarities[0]),
        "top1_ged": top1_ged,
        "min_ged_top5": min_ged_top5,
        "min_ged_top10": min_ged_top10,
        "min_ged_all": min_ged_all,
        "rank_of_min_ged": rank_of_min_ged,
        "best_ged": min_ged_all,
        "best_ged_smiles": best_ged_smiles,
        "top1_tanimoto": top1_tani,
        "tanimoto_top5": [t for t in tani_top5 if t is not None],
        "tanimoto_top10": [t for t in tani_top10 if t is not None],
        "num_valid_candidates": len(valid_ged),
    }
