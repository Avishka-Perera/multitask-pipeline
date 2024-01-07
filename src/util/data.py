import json
import os
import glob
from typing import Sequence
import numpy as np
import re


def extract_parts(input_string):
    match = re.match(r"^(\d+)([a-zA-Z0-9]*)", input_string)

    if match:
        numerical_part = match.group(1)
        alphanumeric_part = match.group(2)
        return int(numerical_part), alphanumeric_part
    else:
        return float("inf"), input_string


def get_break_points(total, fracs):
    assert sum(fracs) >= 0.99
    fracs = np.array(fracs)
    assert all(
        fracs[:-1] >= fracs[1:]
    ), f"The fracs list must have items in the descending order"
    brk_pts = (total * fracs.cumsum()).round()
    for i in range(len(fracs))[::-1]:
        if i == len(fracs) - 1:
            if brk_pts[i] == total:
                brk_pts[i] = total - 1
        else:
            curr_val = brk_pts[i]
            prev_val = brk_pts[i + 1]
            if curr_val >= prev_val:
                brk_pts[i] = prev_val - 1
    brk_pts = list(brk_pts.astype(int) + 1)
    if brk_pts[0] < 0:
        for i in range(len(fracs)):
            if brk_pts[i] < 0:
                brk_pts[i] = 0
            else:
                break
        raise ValueError(brk_pts)
    brk_pts.insert(0, 0)

    return brk_pts


def remove_if_starts_with(txt, startswith):
    if txt.startswith(startswith):
        return txt[len(startswith) :]
    else:
        return txt


def split_class_dataset(
    imgs_dir: str,
    ant_dir: str,
    img_suffixes: Sequence[str] = ["jpg", "JPG", "png", "JPEG"],
    split_rat: Sequence[float] = [0.8, 0.1, 0.1],
    split_names: Sequence[str] = ["train", "val", "test"],
    path_prefix: str = "images",
) -> None:
    """
    Splits a given class dataset and exports the split definitions.
    Requires the following conditions:
        1. All images of a specific class must be in a single directory.
        2. Provided `imgs_dir` must contains all the classes.
        Else considered as invalid
    """
    assert len(split_rat) == len(split_names)
    if not os.path.exists(imgs_dir):
        raise FileNotFoundError(
            f"The specified 'imgs_dir' ({imgs_dir}) does not exist."
        )
    img_paths = []
    for suf in img_suffixes:
        img_paths += [
            os.path.join(
                path_prefix.strip("/"), remove_if_starts_with(p, imgs_dir).lstrip("/")
            )
            for p in glob.glob(f"{imgs_dir}/*/*.{suf}")
        ]

    if len(img_paths) == 0:
        raise FileNotFoundError("The specified directory is empty or invalid")
    if sum(split_rat) < 0.999 or sum(split_rat) > 1.0001:
        raise ValueError(f"Invalid split ratio definition")

    classes = sorted(
        tuple(set([p.split("/")[-2] for p in img_paths])),
        key=lambda it: extract_parts(it),
    )

    split_paths = {nm: [] for nm in split_names}
    for cls in classes:
        cls_imgs = sorted([p for p in img_paths if p.split("/")[-2] == cls])
        try:
            brk_pts = get_break_points(len(cls_imgs), split_rat)
        except ValueError as e:
            if type(e.args[0]) == list:
                print(f"WARNING: Not sufficient samples for class '{cls}'")
                brk_pts = e.args[0]
            else:
                msg = f"{str(e)}: {cls}"
                raise ValueError(msg)

        for i, split_nm in enumerate(split_names):
            s = brk_pts[i]
            e = brk_pts[i + 1]
            if e == 0:
                e = 1
            split_paths[split_nm].extend(cls_imgs[s:e])

    for nm, lst in split_paths.items():
        print(nm, len(lst))

    split_dir = os.path.join(ant_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)
    for nm, lst in split_paths.items():
        if len(lst) > 0:
            save_path = os.path.join(split_dir, f"{nm}.txt")
            txt = "\n".join(lst)
            with open(save_path, "w") as handler:
                handler.write(txt)

    with open(os.path.join(ant_dir, "classes.json"), "w") as handler:
        json.dump({i: c for (i, c) in enumerate(classes)}, handler, indent=4)


class ParallelDataLoader:
    def __init__(self, dataloaders) -> None:
        self.dataloaders = dataloaders
        self.iterators = [iter(dl) for dl in self.dataloaders]

    def __len__(self) -> int:
        return min([len(dl) for dl in self.dataloaders])

    def __iter__(self):
        self.iterators = [iter(dl) for dl in self.dataloaders]
        return self

    def __next__(self):
        try:
            # TODO
            batch = {}
            for loader_iter in self.iterators:
                loader_batch = next(loader_iter)
                batch.update(loader_batch)
            return batch
        except StopIteration:
            self.iterators = [iter(loader) for loader in self.dataloaders]
            raise StopIteration
