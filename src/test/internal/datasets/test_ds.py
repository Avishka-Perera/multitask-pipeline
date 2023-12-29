from torch.utils.data import Dataset
import torch
import numpy as np


class TestDataset(Dataset):
    def __init__(self, root, split, resize_wh) -> None:
        self.root = root
        self.split = split
        self.resize_wh = resize_wh

    def __len__(self):
        return 128

    def __getitem__(self, index):
        img = torch.Tensor(np.random.random((3, *self.resize_wh[::-1])))
        lbl = np.int64(np.random.randint(0, 10))
        return {"img": img, "lbl": lbl}
