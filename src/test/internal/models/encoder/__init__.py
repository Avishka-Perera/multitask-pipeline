from .vit import test as test_vit
from .resnet import test as test_resnet
from .....util import Logger


def test(
    model_dir: str,
    device: int,
    logger: Logger,
) -> None:
    logger.info("Testing Encoders...")
    test_vit(model_dir, device, logger)
    test_resnet(device, logger)
