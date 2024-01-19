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


class ClassificationEvaluator(BaseEvaluator):
    def __init__(
        self,
        out_path: str = None,
        has_aug_ax: bool = False,
    ) -> None:
        self.has_auf_ax = has_aug_ax
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
        self.conf_mat_path = os.path.join(out_path, "confusion-matrix.png")
        self.conf_csv_path = os.path.join(out_path, "confusion-matrix.csv")
        self.report_path = os.path.join(out_path, "report.txt")

    def process_batch(
        self, batch: Dict[str, torch.Tensor], info: Dict[str, torch.Tensor]
    ) -> Dict[str, list]:
        logits = info["logits"]
        labels = batch["lbl"]
        if self.has_auf_ax:
            logits = flatten_leads(logits, 2)
            labels = flatten_leads(labels, 2)
        logits = logits.cpu().numpy()
        labels = labels.cpu().numpy()
        preds = logits.argmax(axis=1)

        return {"preds": preds.tolist(), "labels": labels.tolist()}

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

    def output(self, results: List[Dict[str, list]]) -> str:
        preds = reduce(lambda i, r: i + r["preds"], results, [])
        labels = reduce(lambda i, r: i + r["labels"], results, [])
        self.preds = preds
        self.labels = labels
        self._export()
