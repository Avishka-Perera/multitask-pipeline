from abc import abstractmethod
from typing import Sequence, Dict
from torch import Tensor


class BaseLoss:
    @abstractmethod
    def __init__(self, device: int, *args, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def __call__(
        self,
        info: Dict[str, Tensor],
        batch: Sequence[Tensor],
    ) -> Dict[str, Tensor]:
        raise NotImplementedError()
