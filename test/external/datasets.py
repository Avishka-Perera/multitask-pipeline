from omegaconf import OmegaConf
from mt_pipe.util import Logger, load_class
import numpy as np
from torch.utils.data import Dataset, Subset
import torch


def get_test_subset(ds: Dataset, test_cnt: int, batch_size: int = None) -> Subset:
    test_cnt = max(min(test_cnt, len(ds)), 2)
    if batch_size is not None:
        test_cnt = int(batch_size * (test_cnt // batch_size))
    ids = np.random.randint(0, len(ds), test_cnt)
    ids[0] = 0
    ids[-1] = len(ds) - 1
    ds = Subset(ds, ids)
    return ds


def test(logger: Logger, conf: OmegaConf, test_cnt: int) -> None:
    logger.info("Testing Datasets...")

    for ds_name, conf in conf.items():
        logger.info(f"Testing {ds_name}...")
        cls = load_class(conf.target)
        for split in conf.split:
            ds = cls(root=conf.root, split=split, **dict(conf.params))
            if "sample_conf" in conf:
                ds = get_test_subset(ds, test_cnt)
                for sample in ds:
                    for itm_nm, itm_conf in conf.sample_conf.items():
                        itm = sample[itm_nm]
                        for atr, trgt in itm_conf.items():
                            if atr == "dtype":
                                assert itm.dtype == load_class(trgt)
                            if atr == "shape":
                                assert itm.shape == torch.Size(trgt)
                            if atr == "min":
                                assert itm.min() >= trgt
                            if atr == "max":
                                assert itm.max() <= trgt
