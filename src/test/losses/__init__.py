from .mmcr import test as test_mmcr
from .cross_entropy import test as test_ce
from .concat_loss import test as test_concat
from ...util import Logger


def test(device: int, logger: Logger) -> None:
    logger.info("Testing Costs...")
    test_mmcr(device, logger)
    test_ce(device, logger)
    test_concat(device, logger)
