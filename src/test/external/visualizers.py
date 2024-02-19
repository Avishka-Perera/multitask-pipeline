import os
from omegaconf import OmegaConf
from mt_pipe.src.util import Logger, make_obj_from_conf
from .util import make_random_nested_obj
from torch.utils.tensorboard import SummaryWriter


def test(logger: Logger, conf: OmegaConf, log_dir: str) -> None:
    logger.info("Testing Visualizers...")

    log_dir = os.path.join(log_dir, "visualizers")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    for k, sub_conf in conf.items():
        logger.info(f"Testing {k}...")

        vis_obj = make_obj_from_conf(sub_conf)
        inp = make_random_nested_obj(sub_conf.input_conf)

        # test without setting tensorboard
        vis_obj(**inp)

        # test with setting the tensorboard
        vis_obj.set_writer(writer)
        vis_obj(**inp)
