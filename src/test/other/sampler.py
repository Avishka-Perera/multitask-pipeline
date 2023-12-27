import os
from torch.utils.data import DataLoader
from src.datasets import ClassDataset
from src.other import ClassOrderedDistributedSampler
import math
import numpy as np
from ...util import Logger
from ..datasets.common import get_test_subset


def test(data_dir: str, logger: Logger) -> None:
    logger.info("Testing ClassOrderedDistributedSampler...")

    bs = 16
    subset_sz = 1024
    mock_num_replicas = 2
    mock_rank = 0
    ds = ClassDataset(os.path.join(data_dir, "tea-grade-v2"))
    ds = get_test_subset(ds, subset_sz)

    sampler = ClassOrderedDistributedSampler(ds, mock_num_replicas, mock_rank)
    dl = DataLoader(ds, bs, sampler=sampler)

    for i, batch in enumerate(dl):
        lbls = batch["lbl"].numpy()
        max_pos = np.where(lbls == lbls.max())[0][0]

        exp_arr = np.hstack(
            ([lbls[: max_pos + 1].tolist()] * math.ceil(len(lbls) / max_pos))
        )[: len(lbls)]
        assert (exp_arr != lbls).sum() == 0
        for i in range(max_pos - 1):
            assert lbls[i + 1] > lbls[i]
        break
