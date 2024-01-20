import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from ._base import BaseEvaluator
import pandas as pd
from functools import reduce


class DepthEvaluator(BaseEvaluator):
    def __init__(
        self,
        out_path: str = None,
    ) -> None:
        if out_path is not None:
            self.save_to_disk = True
            os.makedirs(out_path, exist_ok=True)
            self.conf_mat_path = os.path.join(out_path, "confusion-matrix.png")
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
    ) -> Dict[str, list]:
        pred = info["f7"]
        gt = batch["depth_map"]
        pred = pred.cpu()
        gt = gt.cpu()

        return {"pred": pred.tolist(), "gt": gt.tolist()}

    def _get_report(self) -> str:
        acc = (
            torch.exp(
                (-1) * (torch.abs(torch.Tensor(self.gt) - torch.Tensor(self.pred)))
            )
        ).mean()
        report = f"Accuracy: {acc}"
        if self.save_to_disk:
            with open(self.report_path, "w") as handler:
                handler.write(report)
        return report

    def _export(self) -> str:
        report = self._get_report()
        return report

    def output(self, results: List[Dict[str, list]]):
        pred = reduce(lambda i, r: i + r["pred"], results, [])
        gt = reduce(lambda i, r: i + r["gt"], results, [])
        self.pred = pred
        self.gt = gt
        self._export()
