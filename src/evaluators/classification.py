import os
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from ._base import BaseEvaluator
from ..util import flatten_leads
from torch import distributed as dist
import pandas as pd


# TODO: handle the DDP gatherings within the trainer. And get rid of the rank and world size in the __init__()
class ClassificationEvaluator(BaseEvaluator):
    def __init__(
        self,
        out_path: str = None,
        rank: int = None,
        world_size: int = None,
        has_aug_ax: bool = False,
    ) -> None:
        self.has_auf_ax = has_aug_ax
        self.labels = []
        self.preds = []
        if out_path is not None:
            self.save_to_disk = True
            os.makedirs(out_path, exist_ok=True)
            self.conf_mat_path = os.path.join(out_path, "confusion-matrix.png")
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
        self.conf_mat_path = os.path.join(out_path, "confusion-matrix.png")
        self.report_path = os.path.join(out_path, "report.txt")
        self.conf_csv_path = os.path.join(out_path, "conf_mat.csv")

    def register(
        self, batch: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor]
    ) -> None:
        logits = out["logits"]
        labels = batch["lbl"]
        if self.has_auf_ax:
            logits = flatten_leads(logits, 2)
            labels = flatten_leads(labels, 2)
        logits = logits.cpu().numpy()
        labels = labels.cpu().numpy()
        preds = logits.argmax(axis=1)

        self.preds.extend(preds.tolist())
        self.labels.extend(labels.tolist())

    def _plot_confusion_matrix(self) -> None:
        cm = confusion_matrix(self.labels, self.preds)

        # Plot confusion matrix using seaborn
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=np.unique(self.labels),
            yticklabels=np.unique(self.labels),
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        if self.save_to_disk:
            plt.close()
            fig.savefig(self.conf_mat_path)
        else:
            plt.show()
        df = pd.DataFrame(cm)
        df.to_csv(self.conf_csv_path, index=True)

    def _get_report(self) -> str:
        acc = (np.array(self.preds) == np.array(self.labels)).sum() / len(self.preds)
        report = f"Accuracy: {acc}"
        if self.save_to_disk:
            with open(self.report_path, "w") as handler:
                handler.write(report)
        return report

    def _export(self) -> str:
        self._plot_confusion_matrix()
        report = self._get_report()
        return report

    def output(self) -> str:
        if self.is_ddp:
            all_preds = [None for _ in range(self.world_size)]
            if self.rank == 0:
                dist.gather_object(self.preds, all_preds)
            else:
                dist.gather_object(self.preds)
            all_labels = [None for _ in range(self.world_size)]
            if self.rank == 0:
                dist.gather_object(self.labels, all_labels)
            else:
                dist.gather_object(self.labels)

            if self.rank == 0:
                self.preds = [v for lst in all_preds for v in lst]
                self.labels = [v for lst in all_labels for v in lst]
                return self._export()
            else:
                return ""
        else:
            return self._export()
