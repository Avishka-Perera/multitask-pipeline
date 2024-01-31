import os
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from ._base import BaseEvaluator
from ..util import flatten_leads
import pandas as pd
from functools import reduce


class SegmentationEvaluator(BaseEvaluator):
    def __init__(
        self,
        out_path: str = None,
        rank: int = None,
        world_size: int = None,
        has_aug_ax: bool = False,
    ) -> None:
        self.has_auf_ax = has_aug_ax
        self.total_intersec = 0
        self.total_union = 0

        if out_path is not None:
            self.save_to_disk = True
            os.makedirs(out_path, exist_ok=True)
            self.report_path = os.path.join(out_path, "report.txt")
        else:
            self.save_to_disk = False
            self.conf_mat_path = None
            self.report_path = None
        self.rank = rank
        self.world_size = world_size
        self.is_ddp = rank is not None

    def set_out_path(self, out_path: str) -> None:
        self.save_to_disk = True
        os.makedirs(out_path, exist_ok=True)
        self.report_path = os.path.join(out_path, "report.txt")

    def process_batch(
        self, batch: Dict[str, torch.Tensor], info: Dict[str, torch.Tensor]
    ) -> None:
        logits = info["logits"]
        labels = batch["seg"]
        total_intersec, total_union = [], []
        for ind in range(labels.shape[0]):
            intersect, union = self.iou(labels[ind], logits[ind])
            total_intersec += intersect
            total_union += union
        return {"total_intersec":total_intersec, "total_union":total_union}


    def iou(self, true_image, pred_img, threshold=0.5):
        pred_image_ = np.where(pred_img > threshold, 1, 0)

        def calculate_iou(pred_mask, true_mask):
            intersection = torch.logical_and(pred_mask, true_mask)
            union = torch.logical_or(pred_mask, true_mask)
            return torch.sum(intersection), torch.sum(union)

        def calculate_miou(true_image, pred_image):
            pred_image, true_image = torch.tensor(pred_image), torch.tensor(true_image)
            unique_labels = pred_image.shape[0]
            total_intersec = []
            total_union = []

            for i in range(unique_labels):
                intersect, union = calculate_iou(pred_image[i], true_image[i])
                total_intersec += [intersect]
                total_union += [union]
            return total_intersec, total_union
        return calculate_miou(true_image, pred_image_)

    def _get_report(self) -> str:
        acc = np.array(self.total_intersec).mean() / np.array(self.total_union).mean()
        report = f"Accuracy: {acc}"
        if self.save_to_disk:
            with open(self.report_path, "w") as handler:
                handler.write(report)
        return report

    def output(self, results: List[Dict[str, list]]) -> str:
        total_intersec = reduce(lambda i, r: i + r["total_intersec"], results, [])
        total_union = reduce(lambda i, r: i + r["total_union"], results, [])
        self.total_intersec = total_intersec
        self.total_union = total_union
        self._get_report()
