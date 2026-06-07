"""
NT-Xent (Normalised Temperature-scaled Cross Entropy) Loss.

Direct adaptation from MatExpert's retrieval/utils.py for spectral contrastive
learning. Key difference: here we use a single encoder (shared weights for
anchor and positive views), whereas MatExpert uses two separate encoders
(property vs structure). The loss formulation is identical.

Reference: Chen et al., "A Simple Framework for Contrastive Learning of
Visual Representations" (SimCLR), ICML 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalised Temperature-scaled Cross Entropy) loss.

    For a batch of N positive pairs (z_i, z_j), constructs a 2N×2N similarity
    matrix. The positives are the (i, i+N) and (i+N, i) entries; all other
    pairs in the batch serve as negatives.
    """

    def __init__(self, batch_size: int, temperature: float = 0.1):
        """
        Parameters
        ----------
        batch_size : per-GPU batch size (the loss is computed independently
                     per device).
        temperature : scaling factor for logits (default 0.1, matching
                      MatExpert's optimal setting).
        """
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        # Pre-compute negatives mask: 2N × 2N, True where (i, j) is a negative
        # (i.e., not a positive pair — the diagonal of the off-diagonal blocks)
        self.register_buffer(
            "negatives_mask",
            (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float(),
        )

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_i : (B, D) — L2-normalised embeddings of view 1 (anchors).
        z_j : (B, D) — L2-normalised embeddings of view 2 (positives).

        Returns
        -------
        Scalar loss tensor.
        """
        B = z_i.shape[0]
        assert B == self.batch_size, \
            f"Expected batch_size={self.batch_size}, got {B}"

        # Concatenate along batch dimension → 2N × D
        representations = torch.cat([z_i, z_j], dim=0)  # (2B, D)

        # Similarity matrix (cosine — since both are L2-normalised, this is dot product)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),  # (2B, 1, D)
            representations.unsqueeze(0),  # (1, 2B, D)
            dim=2,
        )  # (2B, 2B)

        # Positives: sim(z_i, z_j) on the diagonal of the off-diagonal blocks
        sim_ij = torch.diag(similarity_matrix, B)   # (B,) — z_i matched to z_j
        sim_ji = torch.diag(similarity_matrix, -B)  # (B,) — z_j matched to z_i
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # (2B,)

        # Move mask to same device as input
        device = representations.device
        neg_mask = self.negatives_mask.to(device)

        # Denominator: sum over all negatives for each positive
        nominator = torch.exp(positives / self.temperature)           # (2B,)
        denominator = neg_mask * torch.exp(similarity_matrix / self.temperature)  # (2B, 2B)

        loss_partial = -torch.log(
            nominator / torch.sum(denominator, dim=1)
        )  # (2B,)
        loss = torch.sum(loss_partial) / (2 * B)
        return loss
