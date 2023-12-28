from abc import abstractmethod
from typing import Dict
from torch import Tensor
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    valid_splits = ["train", "val", "test"]

    @abstractmethod
    def __init__(self, root: str, split: str = "train", *args, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, int | Tensor]:
        raise NotImplementedError()
