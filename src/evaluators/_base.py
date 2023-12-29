from abc import abstractmethod
from typing import Dict
from torch import Tensor
from ..util import Logger


class BaseEvaluator:
    @abstractmethod
    def __init__(
        self,
        logger: Logger,
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
    def register(self, batch: Dict[str, Tensor], out: Dict[str, Tensor]) -> None:
        raise NotImplementedError()

    @abstractmethod
    def output(self) -> str:
        raise NotImplementedError()
