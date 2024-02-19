from ...util import Logger, load_class, fix_list_len
from .util import make_random_nested_obj, validate_nested_obj
from omegaconf import OmegaConf
from typing import Sequence


def test(logger: Logger, conf: OmegaConf, devices: Sequence[int]) -> None:
    logger.info("Testing Learners...")
    for ln_nm, ln_conf in conf.items():
        if "encoder" in ln_conf:
            # i.e., a this is a sharing learner
            logger.info(f"Testing {ln_nm} (Sharing configuration)...")

            encoder_cls = load_class(ln_conf.encoder.target)
            params = ln_conf.encoder.params if "params" in ln_conf.encoder else ()
            encoder = encoder_cls(**params)

            ln_params = ln_conf.learner.params if "params" in ln_conf.learner else {}
            ln_params = {"encoder": encoder, **ln_params}
        else:
            logger.info(f"Testing {ln_nm} (Independant configuration)...")

            ln_params = ln_conf.learner.params if "params" in ln_conf.learner else {}

        learner_cls = load_class(ln_conf.learner.target)
        learner = learner_cls(**ln_params)
        if len(devices) < learner.device_count:
            new_devices = fix_list_len(devices, learner.device_count)
            learner.set_devices(new_devices)
        else:
            learner.set_devices(devices)

        batch = make_random_nested_obj(ln_conf.input_conf)
        out = learner(batch=batch)
        valid, msg = validate_nested_obj(out, ln_conf.output_conf)

        assert valid, msg
