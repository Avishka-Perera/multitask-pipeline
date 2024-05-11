from torch.utils.data import Sampler
from typing import Iterator
import numpy as np
from typing import Sized


class CyclicSampler(Sampler):
    """Limits the dataset size per epoch by returning a limited set of indices (possibly shuffled)"""

    def __init__(
        self,
        data_source: Sized | None = None,
        cycle_size: int = None,
        shuffle: bool = True,
        start_seed: int = 42,
    ) -> None:
        super().__init__()
        assert cycle_size is not None
        self.cycle_size = cycle_size
        self.epoch = 0
        self.all_indices = np.concatenate(
            [np.arange(len(data_source)), np.arange(len(data_source) % cycle_size)]
        )
        cyclic_len = len(self.all_indices)
        self.cycle_count = max(cyclic_len // cycle_size, 1)
        self.shuffle = shuffle
        self.start_seed = start_seed
        self.shuffle_now()

    def shuffle_now(self):
        if self.shuffle:
            random_state = np.random.RandomState(seed=self.start_seed)
            random_state.shuffle(self.all_indices)
            self.start_seed += 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if epoch % self.cycle_count == 0:
            self.shuffle_now()

    def __iter__(self) -> Iterator:
        i = self.epoch % self.cycle_count
        return iter(
            self.all_indices[int(i * self.cycle_size) : int((i + 1) * self.cycle_size)]
        )

    def __len__(self) -> int:
        return self.cycle_size
