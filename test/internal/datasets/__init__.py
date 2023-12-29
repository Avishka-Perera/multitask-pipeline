from .concat_set import test as test_concatset
from ....util import Logger


def test(data_dir: str, test_cnt: int, logger: Logger) -> None:
    logger.info("Testing Datasets...")
    test_concatset(data_dir, test_cnt, logger)
