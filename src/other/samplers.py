from torch.utils.data import Dataset, DistributedSampler, Subset
from typing import TypeVar, Iterator
import math
import numpy as np
from src.datasets import ClassDataset
from typing import Dict, List
from ..datasets import ConcatSet


T_co = TypeVar("T_co", covariant=True)


class ClassOrderedDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        def get_info_from_class_ds(ds):
            label_cats = ds.label_cats
            cls_lbls = ds.cls_lbls
            return label_cats, cls_lbls

        def get_info_from_concat_ds(ds):
            cls_lbls = []
            for ds in ds.concatset.datasets:
                assert type(ds) == ClassDataset
                cls_lbls.extend(list(ds.cls_lbls))
            cls_lbls = np.array(cls_lbls)
            label_cats = list(set(cls_lbls))
            return label_cats, cls_lbls

        if type(self.dataset) == ClassDataset:
            label_cats, cls_lbls = get_info_from_class_ds(self.dataset)
        elif type(self.dataset) == ConcatSet:
            label_cats, cls_lbls = get_info_from_concat_ds(self.dataset)
        elif type(self.dataset) == Subset:
            ds = self.dataset.dataset
            if type(ds) == ClassDataset:
                label_cats, cls_lbls = get_info_from_class_ds(ds)
            elif type(ds) == ConcatSet:
                label_cats, cls_lbls = get_info_from_concat_ds(ds)
            else:
                raise ValueError(
                    "Invalid dataset type provided to 'src.other.samplers.ClassOrderedDistributedSampler'"
                )
            cls_lbls = [cls_lbls[id] for id in self.dataset.indices]
            label_cats = list(set(cls_lbls))
        else:
            raise ValueError(
                "Invalid dataset type provided to 'src.other.samplers.ClassOrderedDistributedSampler'"
            )
        self.label_cats = label_cats
        self.cls_lbls = cls_lbls

    def _get_indices(self) -> Dict[int, List[int]]:
        indices = dict()

        for lbl in sorted(self.label_cats):
            ids = np.where(self.cls_lbls == lbl)[0]
            indices[lbl] = ids

        rng = np.random.RandomState(self.epoch + self.seed)
        for key, ids in indices.items():
            rng.shuffle(ids)
            id_sz = len(ids) // self.num_replicas
            if id_sz < self.num_replicas:
                trail = ids
                ids = []
            else:
                start = self.rank * id_sz
                end = (self.rank + 1) * id_sz
                trail = ids[self.num_replicas * id_sz :]
                ids = list(ids[start:end])
            if len(trail) > 0:
                if self.rank < len(trail):
                    ids.append(trail[self.rank])
                else:
                    ids.append(trail[-1])
            indices[key] = ids

        return indices

    def __iter__(self) -> Iterator[T_co]:
        indices = self._get_indices()
        max_ids_sz = max([len(v) for v in indices.values()])
        cls_cnt = len(indices)
        mixed_ids = np.zeros(max_ids_sz * cls_cnt, dtype=int)

        for i, ids in enumerate(indices.values()):
            resized_ids = (ids * (math.ceil(max_ids_sz / len(ids))))[:max_ids_sz]
            mixed_ids[i::cls_cnt] = resized_ids

        return iter(mixed_ids)

    def __len__(self) -> int:
        indices = self._get_indices()
        return max([len(v) for v in indices.values()]) * len(indices)
