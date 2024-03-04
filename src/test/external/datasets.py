from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from ...util import Logger, load_class
from .util import validate_nested_obj
import numpy as np
from torch.utils.data import Dataset, Subset


def get_test_subset(ds: Dataset, test_cnt: int, batch_size: int = None) -> Subset:
    test_cnt = min(test_cnt, len(ds))
    if batch_size is not None:
        test_cnt = int(batch_size * (test_cnt // batch_size))
    ids = np.random.randint(0, len(ds), test_cnt)
    ids[0] = 0
    ids[-1] = len(ds) - 1
    ds = Subset(ds, ids)
    return ds


def test(logger: Logger, conf: OmegaConf, test_cnt: int) -> None:
    logger.info("Testing Datasets...")

    def validate_single(root, split, params, sample_conf):
        ds = cls(root=root, split=split, **params)
        assert len(ds) > 0, "Dataset length cannot be 0"
        if "sample_conf" in conf:
            ds = get_test_subset(ds, test_cnt)
            for sample in ds:
                valid, msg = validate_nested_obj(sample, sample_conf)
                assert valid, msg

    for ds_name, conf in conf.items():
        ds_count = (
            len(conf.splits) * len(conf.root)
            if type(conf.root) == ListConfig
            else len(conf.splits)
        )
        logger.info(f"Testing {ds_name}({ds_count})...")
        cls = load_class(conf.target)
        params = conf.params if "params" in conf else {}
        for split in conf.splits:
            if type(conf.root) == ListConfig:
                for root in conf.root:
                    validate_single(root, split, params, conf.sample_conf)
            else:
                validate_single(conf.root, split, params, conf.sample_conf)
