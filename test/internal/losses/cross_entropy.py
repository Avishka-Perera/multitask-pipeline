import torch
from ....losses import CrossEntropyLoss
from ....util import Logger, is_lists_equal
import numpy as np
from typing import Dict


def validate_losspack(output: Dict[str, torch.Tensor]) -> None:
    assert is_lists_equal(list(output.keys()), ["tot"])
    loss = output["tot"]
    assert loss.shape == torch.Size([])
    assert loss.dtype == torch.float
    assert loss.cpu().detach().item() >= 0 or torch.isnan(loss)


def test(device: int, logger: Logger) -> None:
    logger.info("Testing CrossEntropyLoss...")
    loss_fn_ce = CrossEntropyLoss(device=device)
    loss_fn_ce_aug = CrossEntropyLoss(device=device, has_aug_ax=True)

    B, K, C, W, H = 16, 8, 3, 224, 224
    out_D = 20
    mock_out_ce = {"logits": torch.Tensor(B, out_D)}
    mock_batch_ce = {
        "lbl": torch.Tensor(np.random.randint(0, out_D, B)).to(int),
        "img": torch.Tensor(B, C, W, H),
    }
    mock_out_ce_aug = {"logits": torch.Tensor(B, K, out_D)}
    mock_batch_ce_aug = {
        "lbl": torch.Tensor(np.random.randint(0, out_D, (B, K))).to(int),
        "img": torch.Tensor(B, K, C, W, H),
    }

    loss_ce = loss_fn_ce(mock_out_ce, mock_batch_ce)
    loss_ce_aug = loss_fn_ce_aug(mock_out_ce_aug, mock_batch_ce_aug)

    for loss in [loss_ce, loss_ce_aug]:
        validate_losspack(loss)
    # assert loss_ce.shape == torch.Size([])
    # assert loss_ce_aug.shape == torch.Size([])
    # assert loss_ce.dtype == torch.float
    # assert loss_ce_aug.dtype == torch.float
    # assert loss_ce.cpu().detach().item() >= 0 or torch.isnan(loss_ce)
    # assert loss_ce_aug.cpu().detach().item() >= 0 or torch.isnan(loss_ce_aug)
