from torch import nn
from typing import Sequence, Dict
from .util import load_class
from ..learners import BaseLearner
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


class LearnerMux(nn.Module):
    device_count = 2

    def __init__(
        self,
        devices: Sequence[int],
        chldrn: Dict | DictConfig,  # cannot use children
        encoder: Dict | DictConfig,
    ) -> None:
        super().__init__()
        encoder = OmegaConf.create(encoder)
        backbone_cls = load_class(encoder.target)
        backbone_params = dict(encoder.params) if "params" in encoder else {}
        self.encoder = backbone_cls(**backbone_params).cuda(devices[0])
        chldrn = OmegaConf.create(chldrn)
        for ch_nm, conf in chldrn.items():
            ch_cls = load_class(conf.target)
            ch_obj: BaseLearner = ch_cls(encoder=self.encoder, devices=devices)
            setattr(self, ch_nm, ch_obj)

            # fill missing params with default params (full)
            if "out_map" not in conf:
                conf["out_map"] = {"path": "spread"}
            if "in_map" not in conf:
                conf["in_map"] = {k: "full" for k in conf["out_map"].keys()}
            else:
                for path_name in conf["out_map"].keys():
                    if path_name not in conf["in_map"]:
                        conf["in_map"][path_name] = "full"

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
                if out_nm == "spread":
                    out.update(ch_ln_out)
                else:
                    out[out_nm] = ch_ln_out

        return out
