from .classification import test as test_classification
from ...util import Logger


def test(out_dir: str, logger: Logger) -> None:
    logger.info("Testing Evaluators...")
    test_classification(out_dir, logger)
