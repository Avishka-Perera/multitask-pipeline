import os
from omegaconf import OmegaConf
from ...util import Logger, make_obj_from_conf
from .util import make_random_nested_obj, validate_nested_obj


def test(logger: Logger, conf: OmegaConf):
    logger.info("Testing Augmentors...")

    for nm, sub_conf in conf.items():
        logger.info(f"Testing {nm}...")

        aug = make_obj_from_conf(sub_conf)
        if "pre" in sub_conf:
            input = make_random_nested_obj(sub_conf.pre.input_conf)
            out = aug.pre_collate_routine(input)
            valid, msg = validate_nested_obj(out, sub_conf.pre.output_conf)
            assert valid, msg
        if "post" in sub_conf:
            input = make_random_nested_obj(sub_conf.post.input_conf)
            out = aug.post_collate_routine(input)
            valid, msg = validate_nested_obj(out, sub_conf.post.output_conf)
            assert valid, msg
