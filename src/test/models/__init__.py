from typing import Sequence
from .encoder import test as test_encoder
from ....util import Logger


def test(
    model_dir: str,
    devices: Sequence[int],
    logger: Logger,
) -> None:
    logger.info("Testing Models...")
    test_encoder(model_dir, devices[0], logger)
