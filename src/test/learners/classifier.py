from ...learners import ClassifierLearner
import torch
from ...util import Logger, fix_list_len
from typing import Sequence
from ...constants import img_wh


def test(devices: Sequence[int], logger: Logger) -> None:
    logger.info("Testing ClassifierLearner...")
    if len(devices) != ClassifierLearner.device_count:
        devices = fix_list_len(devices, ClassifierLearner.device_count)
    for enc_name, enc_params in zip(
        ["ResNet50", "ViT-B"],
        [{"drop_out": 0.2}, {"weights_path": "models/mae_pretrain_vit_base.pth"}],
    ):
        out_D = 20
        batch_size = 8
        aug_count = 4
        ln = ClassifierLearner(
            enc_name=enc_name,
            enc_params=enc_params,
            head_trail=[512, out_D],
            dec_params={"use_batch_norm": True, "drop_out": 0.2},
            devices=devices,
            logger=logger,
        )
        mock_batch = {
            "lbl": torch.Tensor(batch_size, aug_count),
            "img": torch.Tensor(batch_size, aug_count, 3, *img_wh[::-1]),
        }
        out = ln(mock_batch)
        embs, logits = out["embs"], out["logits"]
        assert embs.shape == torch.Size([batch_size, aug_count, ln.emb_D])
        assert embs.dtype == torch.float
        assert logits.shape == torch.Size([batch_size, aug_count, ln.out_D])
        assert logits.dtype == torch.float
