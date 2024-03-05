import os
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from ...util import Logger, load_class
from .util import validate_nested_obj
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
from tqdm import tqdm
import traceback


def get_test_subset(ds: Dataset, test_cnt: int, batch_size: int = None) -> Subset:
    test_cnt = min(test_cnt, len(ds))
    if batch_size is not None:
        test_cnt = int(batch_size * (test_cnt // batch_size))
    ids = np.random.randint(0, len(ds), test_cnt)
    ids[0] = 0
    ids[-1] = len(ds) - 1
    ds = Subset(ds, ids)
    return ds


def reshape4batch(conf, batch_size):
    new_conf = OmegaConf.create()
    for k, v in conf.items():
        if k == "shape" and isinstance(v, ListConfig):
            new_conf[k] = [batch_size, *v]
        elif k == "dtype" and v in ["int", "float"]:
            new_conf["dtype"] = "torch.float32" if v == "float" else "torch.int64"
            new_conf["shape"] = [batch_size]
        elif any([isinstance(v, cls) for cls in [int, float, str, ListConfig]]):
            new_conf[k] = v
        else:
            new_conf[k] = reshape4batch(v, batch_size)
    return new_conf


def test(
    logger: Logger,
    conf: OmegaConf,
    test_cnt: int,
    batch_size: int,
    num_workers: int,
    log_dir: str,
) -> None:

    logger.info("Testing Datasets...")

    def validate_single(root, split, params, sample_conf):
        params = {"root": root, "split": split, **params}
        try:
            ds = cls(**params)
            assert len(ds) > 0, "Dataset length cannot be 0"
            if "sample_conf" in conf:
                if batch_size is None:
                    ds = get_test_subset(ds, test_cnt)
                    for sample in ds:
                        valid, msg = validate_nested_obj(sample, sample_conf)
                        assert valid, msg
                else:
                    dl = DataLoader(
                        ds, batch_size, num_workers=num_workers, shuffle=False
                    )
                    batch_conf = reshape4batch(sample_conf, batch_size)
                    dl_len = len(dl)
                    for idx, batch in tqdm(
                        enumerate(dl), desc=f"{root} | {split}", total=dl_len
                    ):
                        if idx == dl_len - 1:
                            batch_conf = reshape4batch(
                                sample_conf, len(ds) % batch_size
                            )
                        valid, msg = validate_nested_obj(batch, batch_conf)
                        assert valid, msg
        except Exception as e:
            if batch_size is not None:

                log_path = os.path.join(log_dir, "data-errors.log")
                os.makedirs(log_dir, exist_ok=True)

                indent = "    "
                log = []
                log.append(
                    f"Dataset: {ds.__class__.__name__}({', '.join([f'{k}={v}' for k, v in params.items()])})"
                )
                log.append(indent + f"Idx: {idx}, Batch size: {batch_size}")
                traceback_str = traceback.format_exc()
                traceback_str = indent + traceback_str.replace("\n", "\n" + indent)
                log.append(traceback_str)
                log.append("\n")
                log = "\n\n".join(log)
                print(log)

                with open(log_path, "a") as handler:
                    handler.write(log)

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
