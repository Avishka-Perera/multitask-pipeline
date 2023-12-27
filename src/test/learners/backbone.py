from ...learners import BackBoneLearner
import torch
from ...util import Logger
from typing import Sequence
from ...constants import img_wh


def test(devices: Sequence[int], logger: Logger) -> None:
    logger.info("Testing BackBoneLearner...")
    for enc_name, enc_params in zip(
        ["ResNet50", "ViT-B"],
        [{"drop_out": 0.2}, {"weights_path": "models/mae_pretrain_vit_base.pth"}],
    ):
        ln = BackBoneLearner(
            enc_name=enc_name,
            enc_params=enc_params,
            devices=devices[:1],
            logger=logger,
        )
        ln.device_count
        B, K, C, H, W = 8, 4, 3, *img_wh[::-1]
        mock_batch = {
            "lbl": torch.Tensor(B),
            "img": torch.Tensor(B, K, C, H, W),
        }
        out = ln(mock_batch)
        embs = out["embs"]
        assert embs.shape == torch.Size([B, K, ln.emb_D])
        assert embs.dtype == torch.float
