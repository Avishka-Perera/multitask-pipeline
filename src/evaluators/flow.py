import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from ._base import BaseEvaluator
import pandas as pd
from functools import reduce


class FlowEvaluator(BaseEvaluator):
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
        self.epe = 0
        self.sample = 0

    def set_out_path(self, out_path: str) -> None:
        self.save_to_disk = True
        os.makedirs(out_path, exist_ok=True)
        self.report_path = os.path.join(out_path, "report.txt")

    def process_batch(
        self, batch: Dict[str, torch.Tensor], info: Dict[str, torch.Tensor]
    ) -> Dict[str, list]:
        pred = info["flow_fwd"]['f7']
        gt = batch["flow_map"]
        self.pred = pred.cpu()
        self.gt = gt.cpu()

        self.epe += torch.norm(torch.Tensor(self.gt) - torch.Tensor(self.pred), p=2).mean()
        self.sample += gt.shape[0]


    def _get_report(self) -> str:
        acc = self.epe / self.sample
        report = f"End-Point_Error: {acc}"
        if self.save_to_disk:
            with open(self.report_path, "w") as handler:
                handler.write(report)
        return report

    def _export(self) -> str:
        report = self._get_report()
        return report

    def output(self, results: List[Dict[str, list]]):
        self._export()
