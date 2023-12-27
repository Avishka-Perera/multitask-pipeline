import torch
from torchvision import transforms
from typing import Sequence, Dict


class Augmentor:
    def __init__(self, crop_wh, same=False) -> None:
        self.flip_transform = transforms.RandomHorizontalFlip(p=1.0)
        self.crop_transform = (
            transforms.CenterCrop(crop_wh) if same else transforms.RandomCrop(crop_wh)
        )
        self.same = same

    def __call__(self, img, k):
        """Returns any number of augmentations"""
        if not self.same:
            rotation = k * 90
            img = transforms.functional.rotate(img, rotation)
            flip_id = (k % 8) // 4
            if flip_id == 1:
                img = self.flip_transform(img)
        img = self.crop_transform(img)

        return img


def aug_collate_fn(
    batch: Sequence[Dict[str, int | torch.Tensor]], aug: Augmentor, aug_count=8
) -> Dict[str, torch.Tensor]:
    if type(batch[0]["lbl"]) == torch.Tensor:
        lbls = torch.stack(
            [torch.stack([sample["lbl"]] * aug_count) for sample in batch]
        ).to(int)
    else:
        lbls = torch.Tensor([[sample["lbl"]] * aug_count for sample in batch]).to(int)

    imgs = [sample["img"] for sample in batch]
    imgs = torch.stack(
        [torch.stack([aug(img, i) for i in range(aug_count)]) for img in imgs]
    )

    return {"lbl": lbls, "img": imgs}
