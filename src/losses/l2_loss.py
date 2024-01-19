from typing import Sequence, Dict
import torch
from torch.nn import MSELoss as TorchMSELoss

class L2Loss:
    def __init__(self, device: int = 1, weight: float = 1) -> None:
        self.device = device
        self.weight = weight
        self.loss_fn = TorchMSELoss()

    def __call__(
        self,
        out: Dict[str, torch.Tensor],
        batch: Sequence[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        predictions = out["flow_fwd"]["f7"].cuda(self.device)
        targets = batch["flow_map"].cuda(self.device)

        loss = self.loss_fn(predictions, targets) * self.weight

        return {"tot": loss}
