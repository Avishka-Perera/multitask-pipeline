from torch.utils.data import ConcatDataset, Dataset
from typing import Sequence
from ..util import load_class
from typing import Dict
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig


class ConcatSet(Dataset):
    valid_splits = ["train", "test", "val"]
    default_comp_conf = OmegaConf.create(
        {
            "reps": 1,
            "split_mix": {"train": ["train"], "val": ["val"], "test": ["test"]},
            "params": {"train": {}, "val": {}, "test": {}},
        }
    )

    def _process_comp_conf(self, comp_conf: OmegaConf | Dict):
        if "params" in comp_conf:
            if any(
                [split not in comp_conf["params"] for split in ["train", "val", "test"]]
            ):
                params = comp_conf.pop("params")
                comp_conf["params"] = {"train": params, "val": params, "test": params}

        comp_conf = OmegaConf.merge(self.default_comp_conf, comp_conf)
        return comp_conf

    def __init__(
        self, root: Sequence[str], split: str = "train", conf: Dict | ListConfig = []
    ) -> None:
        assert conf != []
        assert all(["target" in comp_conf for comp_conf in conf])
        assert isinstance(root, Sequence)
        assert type(root[0]) == str
        assert split in self.valid_splits
        assert len(conf) == len(root)

        conf = OmegaConf.create(
            [self._process_comp_conf(comp_conf) for comp_conf in conf]
        )

        component_datasets = []
        datasets = []
        for i, ds_conf in enumerate(conf):
            ds_class = load_class(ds_conf.target)
            class_splits = []
            for foreign_split in ds_conf.split_mix[split]:
                class_splits.append(
                    ds_class(
                        root=root[i], split=foreign_split, **dict(ds_conf.params[split])
                    )
                )
            if len(class_splits) == 1:
                component_dataset = class_splits[0]
            elif len(class_splits) == 0:
                continue
            else:
                component_dataset = ConcatDataset(class_splits)
            component_datasets.append(component_dataset)
            if split == "train":
                datasets.extend([component_dataset] * ds_conf.reps)
            else:
                datasets.append(component_dataset)

        self.component_datasets = component_datasets
        self.concatset = ConcatDataset(datasets)

    def __len__(self) -> int:
        return len(self.concatset)

    def __getitem__(self, index):
        return self.concatset.__getitem__(index)
