from abc import abstractmethod
from typing import Dict
from torch import Tensor


class BaseEvaluator:
    @abstractmethod
    def __init__(
        self,
        out_path: str = None,
        rank: int = None,
        world_size: int = None,
        *args,
        **kwargs
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def set_out_path(self, out_path: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def process_batch(self, batch: Dict[str, Tensor], info: Dict[str, Tensor]) -> None:
        raise NotImplementedError()

    @abstractmethod
    def output(self) -> str:
        raise NotImplementedError()
