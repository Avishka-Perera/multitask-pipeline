from typing import Sequence, Dict
import torch
from torch.nn import CrossEntropyLoss as TorchCrossEntropyLoss
from ..util import flatten_leads
from ._base import BaseLoss


class CrossEntropyLoss(BaseLoss):
    def __init__(
        self, device: int, weight: float = 1, has_aug_ax: bool = False
    ) -> None:
        self.device = device
        self.has_aug_ax = has_aug_ax
        self.weight = weight
        self.loss_fn = TorchCrossEntropyLoss()

    def __call__(
        self,
        out: Dict[str, torch.Tensor],
        batch: Sequence[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        logits = out["logits"].cuda(self.device)
        labels = batch["lbl"].cuda(self.device)
        if self.has_aug_ax:
            logits = flatten_leads(logits, 2)
            labels = flatten_leads(labels, 2)
        loss = self.loss_fn(logits, labels) * self.weight

        return {"tot": loss}
