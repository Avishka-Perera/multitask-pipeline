import os
import torch
import numpy as np
from typing import Dict, List
from mt_pipe.src.evaluators import BaseEvaluator


class SegmentationEvaluator(BaseEvaluator):
    def __init__(
        self,
        out_path: str = None,
    ) -> None:
        if out_path is not None:
            self.save_to_disk = True
            os.makedirs(out_path, exist_ok=True)
            self.report_path = os.path.join(out_path, "report.txt")
        else:
            self.save_to_disk = False
            self.conf_mat_path = None
            self.report_path = None

    def set_out_path(self, out_path: str) -> None:
        self.save_to_disk = True
        os.makedirs(out_path, exist_ok=True)
        self.report_path = os.path.join(out_path, "report.txt")

    def process_batch(
        self, batch: Dict[str, torch.Tensor], info: Dict[str, torch.Tensor]
    ) -> None:
        logits = info["logits"].cpu().detach().numpy()
        logits = logits.argmax(axis=1)
        labels = batch["seg"].cpu().detach().numpy()
        labels = labels.squeeze()
        total_intersec, total_union = 0, 0
        for ind in np.unique(labels):
            labels_mask = np.zeros(labels.shape)
            labels_mask[labels == ind] = 1
            logits_mask = np.zeros(labels.shape)
            logits_mask[logits == ind] = 1
            intersection_mask = np.logical_and(labels_mask, logits_mask)
            union_mask = np.logical_or(labels_mask, logits_mask)
            intersect = intersection_mask.sum()
            union = union_mask.sum()
            total_intersec += intersect
            total_union += union
        return {"intersec": total_intersec, "union": total_union}

    def _save_report(self, iou: float) -> str:
        report = f"IoU: {iou}"
        if self.save_to_disk:
            with open(self.report_path, "w") as handler:
                handler.write(report)
        return report

    def output(self, results: List[Dict[str, int]]) -> str:
        ious = [res["intersec"] / res["union"] for res in results]
        iou = sum(ious) / len(ious)
        return self._save_report(iou)
