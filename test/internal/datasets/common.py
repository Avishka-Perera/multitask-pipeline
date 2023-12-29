import numpy as np
from torch.utils.data import Subset, Dataset


def get_test_subset(ds: Dataset, test_cnt: int, batch_size: int = None) -> Subset:
    test_cnt = max(min(test_cnt, len(ds)), 2)
    if batch_size is not None:
        test_cnt = int(batch_size * (test_cnt // batch_size))
    ids = np.random.randint(0, len(ds), test_cnt)
    ids[0] = 0
    ids[-1] = len(ds) - 1
    ds = Subset(ds, ids)
    return ds
