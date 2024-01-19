from typing import Sequence, Dict
import torch
from torch.nn import CrossEntropyLoss as TorchCrossEntropyLoss
import numpy as np


def flatten_leads(tens: torch.Tensor, dim_count: int) -> torch.Tensor:
    merge_dims = tens.shape[:dim_count]
    unchn_dims = tens.shape[dim_count:]
    new_shape = [np.prod(merge_dims), *unchn_dims]
    return tens.view(*new_shape)


class CrossEntropyLoss:
    def __init__(
        self, device: int, weight_scale: float = 1, has_aug_ax: bool = False
    ) -> None:
        self.device = device
        self.has_aug_ax = has_aug_ax
        self.weight_scale = weight_scale
        self.loss_fn = TorchCrossEntropyLoss()

    def __call__(
        self,
        info: Dict[str, torch.Tensor],
        batch: Sequence[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        logits = info["logits"].cuda(self.device)
        labels = batch["lbl"].cuda(self.device)
        if self.has_aug_ax:
            logits = flatten_leads(logits, 2)
            labels = flatten_leads(labels, 2)
        loss = self.loss_fn(logits, labels) * self.weight_scale

        return {"tot": loss}
