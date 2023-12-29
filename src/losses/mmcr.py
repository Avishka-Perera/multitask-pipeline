import torch
from typing import Sequence, Dict


class MMCRLoss:
    def __init__(
        self,
        device: int,
        lamb: float,
        weight: float = 1,
    ) -> None:
        self.lamb = lamb
        self.device = device
        self.weight = weight

    def __call__(
        self,
        out: Dict[str, torch.Tensor],
        batch: Sequence[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        embs has the shape B x K x D
            B: batch size
            K: augmentation count
            D: dimension of a single embedding
        """
        embs = out["embs"]
        embs = embs.cuda(self.device)

        K = embs.shape[1]

        # to unit sphere
        z = embs.norm(dim=2, keepdim=True)  # B x K x D

        # centroids
        c = z.mean(dim=-1)  # B x D

        U_z, S_z, V_z = z.svd()
        U_c, S_c, V_c = c.svd()

        centroids_divergence = S_c.sum()
        augmentations_divergence = S_z.sum() / K
        loss = self.lamb * augmentations_divergence + -1 * centroids_divergence

        loss = loss * self.weight

        return {
            "tot": loss,
            "centroids_divergence": centroids_divergence,
            "augmentations_divergence": augmentations_divergence,
        }
