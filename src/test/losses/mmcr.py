import torch
from ...losses import MMCRLoss
from ...util import Logger, is_lists_equal
from typing import Dict


def validate_losspack(output: Dict[str, torch.Tensor]) -> None:
    assert is_lists_equal(
        list(output.keys()), ["tot", "centroids_convergence", "distribution_divergence"]
    )
    for loss in output.values():
        assert loss.shape == torch.Size([])
        assert loss.dtype == torch.float


def test(device: int, logger: Logger) -> None:
    logger.info("Testing MMCRLoss...")
    loss_fn = MMCRLoss(lamb=0.01, device=device)
    mock_batch = {"lbl": torch.Tensor(16, 8), "img": torch.Tensor(16, 8, 3, 224, 224)}
    mock_out = {"embs": torch.Tensor(16, 8, 768)}
    loss_pack = loss_fn(mock_out, mock_batch)

    validate_losspack(loss_pack)
