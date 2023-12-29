import torch
from .....encoders import ResNet50, ResNet18
from .....util import Logger, is_lists_equal


def test(device: int, logger: Logger) -> None:
    logger.info("Testing ResNet...")
    model_classes = [ResNet18, ResNet50]
    emb_Ds = [512, 2048]  # specific for ResNet18 and ResNet50
    for model_class, emb_D in zip(model_classes, emb_Ds):
        model = model_class(drop_out=0.2).to(device)
        B, C, H, W = 16, 3, 224, 224
        mock_batch = torch.Tensor(B, C, H, W).to(device)

        feature_pyramid = model(mock_batch)
        assert is_lists_equal(
            list(feature_pyramid.keys()), ["embs", "f1", "f2", "f3", "f4", "f5"]
        )
        assert all([tens.shape[0] == B for tens in feature_pyramid.values()])
        features = {
            "f1": feature_pyramid["f1"],
            "f2": feature_pyramid["f2"],
            "f3": feature_pyramid["f3"],
            "f4": feature_pyramid["f4"],
            "f5": feature_pyramid["f5"],
        }
        embs = feature_pyramid["embs"]
        assert all([len(tens.shape) == 4 for tens in features.values()])
        assert embs.shape == torch.Size((B, emb_D))
