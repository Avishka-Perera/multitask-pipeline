import torch
import os
from ...constants import img_wh
from .common import get_test_subset
from ...util import Logger
from ...datasets import ConcatSet
import numpy as np

conf = [
    {
        "target": "src.datasets.ClassDataset",
        "reps": 1,
        "split_mix": {"train": ["train"], "val": ["val", "test"]},
        "params": {"resize_wh": [224, 224]},
    },
    {
        "target": "src.datasets.ClassDataset",
        "reps": 1,
        "split_mix": {"train": ["train"], "val": ["val", "test"]},
        "params": {"resize_wh": [224, 224]},
    },
    {
        "target": "src.datasets.ClassDataset",
        "reps": 1,
        "split_mix": {"train": ["train"], "val": ["val", "test"]},
        "params": {"resize_wh": [224, 224]},
    },
]


def test(data_dir: str, test_cnt: int, logger: Logger) -> None:
    logger.info("Testing ConcatSet...")
    root = [
        os.path.join(data_dir, "tea-grade-v2"),
        os.path.join(data_dir, "tea-std"),
        os.path.join(data_dir, "plant-doc"),
    ]
    for split in ["val", "test", "train"]:
        ds = ConcatSet(root, split, conf)
        ds = get_test_subset(ds, test_cnt)
        for pair in ds:
            lbl, img = pair["lbl"], pair["img"]
            assert img.shape == (3, *img_wh[::-1]), img.shape
            assert img.dtype == torch.float32
            assert type(lbl) == np.int64
            assert img.min() >= 0
            assert img.max() <= 1
