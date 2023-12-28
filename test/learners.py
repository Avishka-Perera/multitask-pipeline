import torch
from mt_pipe.util import Logger, load_class
from omegaconf import OmegaConf
from typing import Sequence


def test(logger: Logger, conf: OmegaConf, devices: Sequence[int]) -> None:
    logger.info("Testing Learners...")
    for ln_nm, ln_conf in conf.items():
        logger.info(f"Testing {ln_nm}...")
        ln_cls = load_class(ln_conf.target)
        ln = ln_cls(**dict(ln_conf.params), logger=logger, devices=devices)
        batch = {k: torch.Tensor(*v) for k, v in ln_conf.batch_conf.items()}
        out = ln(batch)
        for k, val_conf in ln_conf.val_conf.items():
            for atr, trgt in val_conf.items():
                if atr == "shape":
                    assert out[k].shape == torch.Size(trgt)
                if atr == "dtype":
                    assert out[k].dtype == load_class(trgt)
