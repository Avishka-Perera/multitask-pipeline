from torch.nn import Module
from typing import Dict


class LossMux(Module):
    def __init__(self, losses: Dict) -> None:
        super().__init__()
        self.loss_fns = losses

    def forward(self, info, batch) -> Dict:
        loss_pack = {"tot": 0}
        for dp, loss_fn in self.loss_fns.items():
            loss_pack[dp] = loss_fn(info=info[dp], batch=batch[dp])
            loss_pack["tot"] += loss_pack[dp]["tot"]
        return loss_pack
