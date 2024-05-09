from torch.utils.data import ConcatDataset, Dataset
import importlib
from typing import Dict, Sequence
from omegaconf import OmegaConf, DictConfig, ListConfig

# from omegaconf.listconfig import ListConfig


def load_class(target):
    """loads a class using a target"""
    *module_name, class_name = target.split(".")
    module_name = ".".join(module_name)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


def make_obj_from_conf(conf, **kwargs):
    if isinstance(conf, DictConfig):
        conf = OmegaConf.to_container(conf)
    cls = load_class(conf["target"])
    params = conf["params"] if "params" in conf else {}
    obj = cls(**params, **kwargs)
    return obj


class ConcatSet(Dataset):
    def __init__(self, conf: Sequence[Dict] | ListConfig = []) -> None:
        assert conf != []
        assert all(["target" in comp_conf for comp_conf in conf])

        conf = OmegaConf.create(conf)

        datasets = []
        for ds_conf in conf:
            ds = make_obj_from_conf(ds_conf)
            if "reps" in ds_conf:
                reps = ds_conf.reps
            else:
                reps = 1
            datasets.extend([ds] * reps)

        self.dataset = ConcatDataset(datasets)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)
