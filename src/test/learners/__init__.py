from .backbone import test as test_backbone
from .classifier import test as test_classifier
from typing import Sequence
from ...util import Logger


def test(devices: Sequence[int], logger: Logger) -> None:
    logger.info("Testing Learners...")
    test_backbone(devices, logger)
    test_classifier(devices, logger)
