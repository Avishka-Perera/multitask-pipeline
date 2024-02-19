from omegaconf import OmegaConf
from mt_pipe.src.util import Logger, load_class
from .util import make_random_nested_obj, validate_nested_obj, are_shapes_equal


def test(logger: Logger, conf: OmegaConf) -> None:
    logger.info("Testing Models...")

    for k, sub_conf in conf.items():
        logger.info(f"Testing {k}...")

        model_cls = load_class(sub_conf.target)
        params = sub_conf.params if "params" in sub_conf else {}
        model = model_cls(**params)

        # validate I/O
        input = make_random_nested_obj(sub_conf.input_conf)
        out = model(**input)
        valid, msg = validate_nested_obj(out, sub_conf.output_conf)
        assert valid, msg

        # validate attributes
        if "expected_attributes" in sub_conf:
            for nm, exp_attr in sub_conf.expected_attributes.items():
                assert are_shapes_equal(exp_attr, getattr(model, nm)), (
                    exp_attr,
                    getattr(model, nm),
                )
