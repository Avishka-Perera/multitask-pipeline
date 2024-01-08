from torch import nn
from typing import Sequence, Dict
from .util import load_class
from mt_pipe.src.learners import BaseLearner
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


class LearnerMux(nn.Module):
    device_count = 2

    def __init__(
        self,
        devices: Sequence[int],
        chldrn: Dict | DictConfig,  # cannot use children
        backbone: Dict | DictConfig,
    ) -> None:
        super().__init__()
        backbone = OmegaConf.create(backbone)
        backbone_cls = load_class(backbone.target)
        backbone_params = dict(backbone.params) if "params" in backbone else {}
        self.encoder = backbone_cls(**backbone_params).cuda(devices[0])
        chldrn = OmegaConf.create(chldrn)
        for ch_nm, conf in chldrn.items():
            ch_cls = load_class(conf.target)
            ch_obj: BaseLearner = ch_cls(backbone=self.encoder, devices=devices)
            setattr(self, ch_nm, ch_obj)
        self.chldrn = chldrn

    def forward(self, batch):
        out = {}

        for ch_nm, conf in self.chldrn.items():
            ch_ln = getattr(self, ch_nm)
            for in_map, out_nm in zip(conf.in_map.values(), conf.out_map.values()):
                if in_map == "full":
                    ch_ln_inp = batch
                else:
                    ch_ln_inp = {}
                    for src, dst in in_map.items():
                        ch_ln_inp[dst] = batch[src]
                ch_ln_out = ch_ln(ch_ln_inp)
                if out_nm == ".":
                    return ch_ln_out
                out[out_nm] = ch_ln_out

        return out
