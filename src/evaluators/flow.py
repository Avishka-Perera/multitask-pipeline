import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from ._base import BaseEvaluator
import pandas as pd
from functools import reduce


class FlowMetrics(nn.Module):
    def __init__(self):
        super().__init__()
        self.bad_ths = [0.5, 1, 3, 5]
        self.speed_ths = [(0, 10), (10, 40), (40, torch.inf)]

    def reset(self):
        self.agg_N = 0  # number of pixels so far
        self.agg_L1err = torch.tensor(0.0)  # L1 error so far
        self.agg_L2err = torch.tensor(0.0)  # L2 (=EPE) error so far
        self.agg_Nbad = [0 for _ in self.bad_ths]  # counter of bad pixels
        self.agg_EPEspeed = [
            torch.tensor(0.0) for _ in self.speed_ths
        ]  # EPE per speed bin so far
        self.agg_Nspeed = [0 for _ in self.speed_ths]  # N pixels per speed bin so far
        self._metrics = None
        self.pairname_results = {}

    def add_batch(self, info, batch):
        predictions = info["flow_fwd"]["l7"].detach()
        gt = batch["flow_map"].detach()
        assert predictions.size(1) == 2, predictions.size()
        assert gt.size(1) == 2, gt.size()
        if (
            gt.size(2) == predictions.size(2) * 2
            and gt.size(3) == predictions.size(3) * 2
        ):  # special case for Spring ...
            L1err = torch.minimum(
                torch.minimum(
                    torch.minimum(
                        torch.sum(torch.abs(gt[:, :, 0::2, 0::2] - predictions), dim=1),
                        torch.sum(torch.abs(gt[:, :, 1::2, 0::2] - predictions), dim=1),
                    ),
                    torch.sum(torch.abs(gt[:, :, 0::2, 1::2] - predictions), dim=1),
                ),
                torch.sum(torch.abs(gt[:, :, 1::2, 1::2] - predictions), dim=1),
            )
            L2err = torch.minimum(
                torch.minimum(
                    torch.minimum(
                        torch.sqrt(
                            torch.sum(
                                torch.square(gt[:, :, 0::2, 0::2] - predictions), dim=1
                            )
                        ),
                        torch.sqrt(
                            torch.sum(
                                torch.square(gt[:, :, 1::2, 0::2] - predictions), dim=1
                            )
                        ),
                    ),
                    torch.sqrt(
                        torch.sum(
                            torch.square(gt[:, :, 0::2, 1::2] - predictions), dim=1
                        )
                    ),
                ),
                torch.sqrt(
                    torch.sum(torch.square(gt[:, :, 1::2, 1::2] - predictions), dim=1)
                ),
            )
            valid = torch.isfinite(L1err)
            gtspeed = (
                torch.sqrt(torch.sum(torch.square(gt[:, :, 0::2, 0::2]), dim=1))
                + torch.sqrt(torch.sum(torch.square(gt[:, :, 0::2, 1::2]), dim=1))
                + torch.sqrt(torch.sum(torch.square(gt[:, :, 1::2, 0::2]), dim=1))
                + torch.sqrt(torch.sum(torch.square(gt[:, :, 1::2, 1::2]), dim=1))
            ) / 4.0  # let's just average them
        else:
            valid = torch.isfinite(gt[:, 0, :, :])  # both x and y would be infinite
            L1err = torch.sum(torch.abs(gt - predictions), dim=1)
            L2err = torch.sqrt(torch.sum(torch.square(gt - predictions), dim=1))
            gtspeed = torch.sqrt(torch.sum(torch.square(gt), dim=1))
        N = valid.sum()
        Nnew = self.agg_N + N
        self.agg_L1err = (
            float(self.agg_N) / Nnew * self.agg_L1err
            + L1err[valid].mean().cpu() * float(N) / Nnew
        )
        self.agg_L2err = (
            float(self.agg_N) / Nnew * self.agg_L2err
            + L2err[valid].mean().cpu() * float(N) / Nnew
        )
        self.agg_N = Nnew
        for i, th in enumerate(self.bad_ths):
            self.agg_Nbad[i] += (L2err[valid] > th).sum().cpu()
        for i, (th1, th2) in enumerate(self.speed_ths):
            vv = (gtspeed[valid] >= th1) * (gtspeed[valid] < th2)
            iNspeed = vv.sum()
            if iNspeed == 0:
                continue
            iNnew = self.agg_Nspeed[i] + iNspeed
            self.agg_EPEspeed[i] = (
                float(self.agg_Nspeed[i]) / iNnew * self.agg_EPEspeed[i]
                + float(iNspeed) / iNnew * L2err[valid][vv].mean().cpu()
            )
            self.agg_Nspeed[i] = iNnew

    def _compute_metrics(self):
        if self._metrics is not None:
            return
        out = {}
        out["L1err"] = self.agg_L1err.item()
        out["EPE"] = self.agg_L2err.item()
        for i, th in enumerate(self.bad_ths):
            out["bad@{:.1f}".format(th)] = (
                float(self.agg_Nbad[i]) / self.agg_N
            ).item() * 100.0
        for i, (th1, th2) in enumerate(self.speed_ths):
            out[
                "s{:d}{:s}".format(th1, "-" + str(th2) if th2 < torch.inf else "+")
            ] = self.agg_EPEspeed[i].item()
        self._metrics = out

    def get_results(self):
        self._compute_metrics()  # to avoid recompute them multiple times
        return self._metrics


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

        self.metrics = FlowMetrics()

    def set_out_path(self, out_path: str) -> None:
        self.save_to_disk = True
        os.makedirs(out_path, exist_ok=True)
        self.report_path = os.path.join(out_path, "report.txt")

    def process_batch(
        self, batch: Dict[str, torch.Tensor], info: Dict[str, torch.Tensor]
    ) -> Dict[str, list]:
        self.metrics.add_batch(info, batch)

    def _get_report(self) -> str:
        acc = self.metrics.get_results()["EPE"]
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
