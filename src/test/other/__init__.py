from .sampler import test as test_sampler
from ...util import Logger


def test(data_dir: str, logger: Logger) -> None:
    logger.info("Testing Other objects...")
    test_sampler(data_dir, logger)
