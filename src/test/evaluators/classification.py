from src.evaluators import ClassificationEvaluator
from ...util import Logger
import torch
import numpy as np


def test(out_dir: str, logger: Logger):
    logger.info("Testing ClassificationEvaluator...")

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
    mock_batch_count = 10
    mock_outs_lst = [
        [mock_out_ce] * mock_batch_count,
        [mock_out_ce_aug] * mock_batch_count,
    ]
    mock_batches_lst = [
        [mock_batch_ce] * mock_batch_count,
        [mock_batch_ce_aug] * mock_batch_count,
    ]
    evs = [
        ClassificationEvaluator(f"{out_dir}/vis"),
        ClassificationEvaluator(f"{out_dir}/vis", has_aug_ax=True),
    ]

    for ev, mock_outs, mock_batches in zip(evs, mock_outs_lst, mock_batches_lst):
        for out, batch in zip(mock_outs, mock_batches):
            ev.register(batch=batch, out=out)

        report = ev.output()
        assert type(report) == str
