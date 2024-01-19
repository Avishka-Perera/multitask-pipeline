from ...util import Logger, load_class
from .util import make_random_nested_tens, validate_nested_obj
from omegaconf import OmegaConf


def test(logger: Logger, conf: OmegaConf, device: int) -> None:
    logger.info("Testing Losses...")

    for loss_nm, loss_conf in conf.items():
        logger.info(f"Testing {loss_nm}...")

        loss_cls = load_class(loss_conf.loss_fn.target)
        params = loss_conf.loss_fn.params if "params" in loss_conf.loss_fn else {}
        loss_fn = loss_cls(**params, device=device)

        info = make_random_nested_tens(loss_conf.input_conf.info)
        batch = make_random_nested_tens(loss_conf.input_conf.batch)

        loss_pack = loss_fn(info=info, batch=batch)
        none_mask = (
            loss_conf.output_conf.none_mask
            if "none_mask" in loss_conf.output_conf
            else None
        )
        valid, msg = validate_nested_obj(
            loss_pack, loss_conf.output_conf.main, none_mask
        )

        assert valid, msg
