import torch
from omegaconf.dictconfig import DictConfig
from typing import Dict, Sequence
from ..util import load_class
from ._base import BaseLoss


class ConcatLoss(BaseLoss):
    def __init__(
        self,
        device: int,
        conf: DictConfig | Dict[str, object] = {},
        weight: float = -1,
    ) -> None:
        self.loss_fns = {}
        first_val = tuple(conf.values())[0]
        if type(first_val) == DictConfig:
            self.device = device
        else:
            self.device = first_val.device
        for name, value in conf.items():
            if type(value) == DictConfig:
                loss_class = load_class(value.target)
                loss_params = dict(value.params)
                loss_fn = loss_class(device=device, **loss_params)
            else:
                loss_fn = value
            self.loss_fns[name] = loss_fn

    def __call__(
        self,
        out: Dict[str, torch.Tensor],
        batch: Sequence[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        total_loss = torch.tensor(0.0).cuda(self.device)
        glob_loss_pack = {}
        for nm, loss_fn in self.loss_fns.items():
            loss_pack = loss_fn(out=out, batch=batch)
            glob_loss_pack[nm] = loss_pack
            loss = loss_pack["tot"]
            if not torch.isnan(loss):
                total_loss += loss.cuda(self.device)
        glob_loss_pack["tot"] = total_loss

        return glob_loss_pack
