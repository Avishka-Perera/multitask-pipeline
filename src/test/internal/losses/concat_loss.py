from omegaconf import OmegaConf
from ....util import Logger, are_lists_equal
import torch
import numpy as np
from ....losses import ConcatLoss, MMCRLoss, CrossEntropyLoss
from .mmcr import validate_losspack as validate_mmcr_losspack
from .cross_entropy import validate_losspack as validate_ce_losspack


def test(device: int, logger: Logger) -> None:
    logger.info("Testing ConcatLoss...")

    conf = OmegaConf.create(
        {
            "mmcr": {"target": "src.losses.MMCRLoss", "params": {"lamb": 1}},
            "cross_entropy": {
                "target": "src.losses.CrossEntropyLoss",
                "params": {"has_aug_ax": True},
            },
        }
    )

    conc_loss_fn = ConcatLoss(device, conf)

    B, K, C, W, H = 16, 8, 3, 224, 224
    emb_D, out_D = 768, 20
    mock_batch = {
        "lbl": torch.Tensor(np.random.randint(0, out_D, (B, K))).to(int),
        "img": torch.Tensor(B, K, C, H, W),
    }
    mock_out = {
        "embs": torch.Tensor(B, K, emb_D),
        "logits": torch.Tensor(B, K, out_D),
    }
    loss_pack = conc_loss_fn(mock_out, mock_batch)
    assert are_lists_equal(list(loss_pack.keys()), ["tot", "mmcr", "cross_entropy"])
    validate_mmcr_losspack(loss_pack["mmcr"])
    validate_ce_losspack(loss_pack["cross_entropy"])
