import os
from omegaconf import OmegaConf
from mt_pipe.src.util import Logger, make_obj_from_conf
from .util import make_random_nested_obj


def test(logger: Logger, conf: OmegaConf, log_dir: str) -> None:
    logger.info("Testing Evaluators...")

    log_dir = os.path.join(log_dir, "evaluators")

    for k, sub_conf in conf.items():
        logger.info(f"Testing {k}...")
        eval_path = os.path.join(log_dir, k)
        os.makedirs(eval_path, exist_ok=True)

        ev_obj = make_obj_from_conf(sub_conf)
        ev_obj.set_out_path(eval_path)
        mock_dl = [make_random_nested_obj(sub_conf.input_conf) for _ in range(5)]

        results = []
        for batch in mock_dl:
            results.append(
                ev_obj.process_batch(batch=batch["batch"], info=batch["info"])
            )
        ev_obj.output(results)
