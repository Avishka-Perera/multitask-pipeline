import numpy as np
import os
import json
from typing import Tuple, Dict
import torch
from PIL import Image
from torchvision import transforms
from ...util.exceptions import DatasetCorruptionError
from .._base import BaseDataset
import logging
from .util import make_s3_ds, make_pcam_ds

logger = logging.getLogger()


class ClassDataset(BaseDataset):
    valid_splits = ["train", "val", "test"]
    s3_uris = {
        "tea-grade-v2": "s3://teaai-datasets/tea-grade/tea-grade-v2/",
        "tea-std": "s3://teaai-datasets/tea-std/",
        "new-plant-disease": "s3://teaai-datasets/other/new-plant-disease/",
        "plant-doc": "s3://teaai-datasets/other/plant-doc/",
        "uc-mlr-leaf": "s3://teaai-datasets/other/uc-mlr-leaf/",
    }
    other_urls = {
        "pcam": "https://drive.google.com/drive/folders/1gHou49cA1s5vua2V5L98Lt8TiWA3FrKB"
    }

    def _make_ds(self, root, dataset) -> None:
        if dataset in self.s3_uris.keys():
            make_s3_ds(self.s3_uris[dataset], root)
        elif dataset == "pcam":
            make_pcam_ds(self.other_urls["pcam"], root)
        else:
            raise ValueError("Unsupported dataset.")

    def __init__(
        self,
        root: str,
        split: str = "train",
        resize_wh: Tuple[int] = None,
        dataset: str = None,
    ) -> None:
        # validate
        if split not in self.valid_splits:
            raise ValueError(
                f"Invalid split definition. Allowed splits are {self.valid_splits}"
            )
        if dataset is not None:
            if dataset not in list(self.s3_uris.keys()) + list(self.other_urls.keys()):
                raise ValueError(
                    f"Invalid dataset definition. Allowed datasets are {list(self.s3_uris.keys())}"
                )
            if not os.path.exists(root):
                self._make_ds(root, dataset)

        # find paths
        with open(
            os.path.join(
                root, "annotations", "classification", "splits", f"{split}.txt"
            )
        ) as handler:
            paths = [os.path.join(root, p) for p in handler.read().strip().split("\n")]

        classes = [p.split("/")[-2] for p in paths]
        with open(
            os.path.join(root, "annotations", "classification", "classes.json")
        ) as handler:
            lbl_cls_map = {int(i): c for (i, c) in json.load(handler).items()}
        cls_lbl_map = {c: i for (i, c) in lbl_cls_map.items()}
        cls_lbls = [cls_lbl_map[c] for c in classes]

        # validate the dataset
        path_classes = list(set(classes))
        for c in lbl_cls_map.values():
            if c in path_classes:
                path_classes.remove(c)
            else:
                logger.warn(f"Samples from class '{c}' are unavailable")
        if len(path_classes) != 0:
            raise DatasetCorruptionError(
                "Defined classes in the `classes.json` does not match with the actual directory names"
            )

        trans_lst = [transforms.ToTensor()]
        if resize_wh is not None:
            trans_lst.append(transforms.Resize(resize_wh, antialias=True))
        self.transforms = transforms.Compose(trans_lst)

        self.paths = paths
        self.class_cats = np.array(sorted(list(set(classes))))
        self.label_cats = np.array(sorted(list(set(cls_lbls))))
        self.cls_lbl_map = cls_lbl_map
        self.lbl_cls_map = lbl_cls_map
        self.cls_lbls = np.array(cls_lbls)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index) -> Dict[str, int | torch.Tensor]:
        path = self.paths[index]
        img = Image.open(path).convert("RGB")
        img = torch.clip(self.transforms(img), 0, 1)
        lbl = self.cls_lbls[index]
        return {"lbl": lbl, "img": img}
