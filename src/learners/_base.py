from abc import abstractmethod
from typing import Dict, Sequence
from torch import Tensor
from torch import nn


class BaseLearner(nn.Module):
    valid_splits = ["train", "val", "test"]
    device_count: int = None

    @abstractmethod
    def __init__(self, devices: Sequence[int], *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError()
