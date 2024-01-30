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
        chldrn: Dict | DictConfig,  # cannot use children
        encoder: Dict | DictConfig,
    ) -> None:
        super().__init__()
        encoder = OmegaConf.create(encoder)
        backbone_cls = load_class(encoder.target)
        backbone_params = dict(encoder.params) if "params" in encoder else {}
        self.encoder = backbone_cls(**backbone_params)
        chldrn = OmegaConf.create(chldrn)
        child_names = []
        for ch_nm, conf in chldrn.items():
            child_names.append(ch_nm)
            ch_cls = load_class(conf.target)
            params = conf.params if "params" in conf else {}
            ch_obj: BaseLearner = ch_cls(encoder=self.encoder, **params)
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
        self.child_names = child_names

    def set_devices(self, devices: Sequence[int] | Dict[str, Sequence[int]]) -> None:
        self.devices = devices
        if type(devices) == dict:
            for ch_nm, dvs in devices.items():
                ln = getattr(self, ch_nm)
                ln.set_devices(dvs)
        else:
            for ch_nm in self.child_names:
                ln = getattr(self, ch_nm)
                ln.set_devices(devices)

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
