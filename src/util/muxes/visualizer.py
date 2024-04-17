from typing import Dict
from torch import Tensor
from ...visualizers import BaseVisualizer


class VisualizerMux(BaseVisualizer):
    def __init__(self, visualizers):
        self.visualizers = visualizers

    def __call__(
        self, info: Dict[str, Tensor], batch: Dict[str, Tensor], epoch: int, loop: str
    ) -> None:
        for dp, ch_visu in self.visualizers.items():
            ch_visu(info[dp], batch[dp], epoch, loop)
