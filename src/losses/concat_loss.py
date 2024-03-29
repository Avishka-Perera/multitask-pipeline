import torch
from omegaconf.dictconfig import DictConfig
from omegaconf import OmegaConf
from typing import Dict, Sequence
from ..util import load_class


class ConcatLoss:
    def __init__(
        self,
        device: Dict[str, int],
        conf: DictConfig | Dict[str, object] = {},
    ) -> None:
        self.loss_fns = {}
        conf = OmegaConf.create(conf)
        self.conf = conf
        self.devices = device
        self.device = tuple(device.values())[0]
        for name, value in conf.items():
            if type(value) == DictConfig:
                loss_class = load_class(value.target)
                loss_params = dict(value.params) if "params" in value else {}
                loss_fn = loss_class(
                    device=device if type(device) == int else device[name],
                    **loss_params
                )
            else:
                loss_fn = value
            self.loss_fns[name] = loss_fn

    def __call__(
        self,
        info: Dict[str, torch.Tensor],
        batch: Sequence[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        total_loss = torch.tensor(0.0).cuda(self.device)
        glob_loss_pack = {}
        for nm, loss_fn in self.loss_fns.items():
            fn_inp = info[self.conf[nm].branch] if "branch" in self.conf[nm] else info
            loss_pack = loss_fn(info=fn_inp, batch=batch)
            glob_loss_pack[nm] = loss_pack
            loss = loss_pack["tot"]
            if not torch.isnan(loss):
                total_loss += loss.cuda(self.device)
        glob_loss_pack["tot"] = total_loss

        return glob_loss_pack
