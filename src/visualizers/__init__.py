import numpy as np
from abc import abstractmethod
from typing import Dict, Sequence
import random
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class BaseVisualizer:
    def __init__(self, name: str = None, max_imgs_per_batch: int = None) -> None:
        self.name = self.__class__.__name__ if name is None else name
        self.writer: SummaryWriter = None
        self.max_imgs_per_batch = max_imgs_per_batch
        self.global_step = 0

    def set_writer(self, writer: SummaryWriter) -> None:
        self.writer = writer

    def _get_samples(self, input_count: int) -> Sequence[int]:
        if (
            self.max_imgs_per_batch is not None
            and input_count > self.max_imgs_per_batch
        ):
            samples = list(range(input_count))
            random.shuffle(samples)
            samples = samples[: self.max_imgs_per_batch]
            return sorted(samples)
        else:
            return range(input_count)

    @abstractmethod
    def __call__(
        self,
        info: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        epoch: int,
        loop: str,
    ) -> None:
        raise NotImplementedError()

    def _output(self, visualization: np.ndarray, loop: str = None) -> None:
        if self.writer is None:
            plt.imshow(visualization)
            plt.show()
        else:
            self.writer.add_images(
                self.name if loop is None else f"{self.name}/{loop}",
                visualization,
                self.global_step,
                dataformats="HWC",
            )
            self.global_step += 1


__all__ = ["BaseVisualizer"]
