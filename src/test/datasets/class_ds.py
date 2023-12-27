import torch
import os
from torch.utils.data import DataLoader
from functools import partial
from ...datasets import ClassDataset
from ...constants import img_wh
from .common import get_test_subset
from ...util import Logger
from ...datasets.collate_fns import aug_collate_fn, Augmentor
import numpy as np


def test_ClassDataset(data_dir: str, test_cnt: int, logger: Logger) -> None:
    logger.info("Testing ClassDataset...")
    for ds_name in [
        "tea-grade-v2",
        "tea-std",
        "new-plant-disease",
        "plant-doc",
        "uc-mlr-leaf",
        "pcam",
        "plant-doc",
    ]:
        for split in ["val", "test", "train"]:
            ds = ClassDataset(
                os.path.join(data_dir, ds_name),
                split,
                resize_wh=img_wh,
                dataset=ds_name,
            )
            ds = get_test_subset(ds, test_cnt)
            for pair in ds:
                lbl, img = pair["lbl"], pair["img"]
                assert img.shape == (3, *img_wh[::-1]), img.shape
                assert img.dtype == torch.float32
                assert type(lbl) == np.int64
                assert img.min() >= 0
                assert img.max() <= 1


def test_ClassDataset_aug(data_dir: str, test_cnt: int, logger: Logger) -> None:
    logger.info("Testing ClassDataset with augmentations...")
    batch_size = 16
    aug_count = 8
    for ds_name in ["tea-grade-v2", "tea-std", "pcam", "plant-doc"]:
        for split in ["val", "test", "train"]:
            ds = ClassDataset(
                os.path.join(data_dir, ds_name), split, img_wh, dataset=ds_name
            )
            ds = get_test_subset(ds, test_cnt * batch_size, batch_size)
            aug = Augmentor(img_wh)
            dl = DataLoader(
                ds,
                batch_size,
                collate_fn=partial(aug_collate_fn, aug_count=aug_count, aug=aug),
            )
            for batch in dl:
                lbls, imgs = batch["lbl"], batch["img"]
                assert imgs.shape == torch.Size(
                    (batch_size, aug_count, 3, *img_wh[::-1])
                ), imgs.shape
                assert imgs.dtype == torch.float
                assert lbls.dtype == torch.int64
                assert lbls.shape == torch.Size([batch_size, aug_count])
                assert imgs.min() >= 0
                assert imgs.max() <= 1
