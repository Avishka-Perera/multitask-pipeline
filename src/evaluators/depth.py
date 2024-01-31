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

            self.report_path = os.path.join(out_path, "report.txt")
        else:
            self.save_to_disk = False
            self.conf_mat_path = None
            self.report_path = None
        self.min_depth = 1e-3
        self.max_depth = 80

    def set_out_path(self, out_path: str) -> None:
        self.save_to_disk = True
        os.makedirs(out_path, exist_ok=True)
        self.report_path = os.path.join(out_path, "report.txt")

    def process_depth(self, gt_depth, pred_depth, min_depth, max_depth):
        mask = gt_depth > 0
        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth
        gt_depth[gt_depth < min_depth] = min_depth
        gt_depth[gt_depth > max_depth] = max_depth

        return gt_depth, pred_depth, mask
    
    # Adopted from https://github.com/mrharicot/monodepth
    def compute_errors(self, gt, pred, nyu=False):
        thresh = np.maximum((gt / pred), (pred / gt))
        if len(thresh) > 0:
            a1 = (thresh < 1.25).mean()
            a2 = (thresh < 1.25**2).mean()
            a3 = (thresh < 1.25**3).mean()
        else:
            a1, a2, a3 = 0, 0, 0
        rmse = (gt - pred)**2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred))**2
        rmse_log = np.sqrt(rmse_log.mean())

        log10 = np.mean(np.abs((np.log10(gt) - np.log10(pred))))

        abs_rel = np.mean(np.abs(gt - pred) / (gt))

        sq_rel = np.mean(((gt - pred)**2) / (gt))

        if nyu:
            return abs_rel, sq_rel, rmse, log10, a1, a2, a3
        else:
            return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


    def process_batch(
        self, batch: Dict[str, torch.Tensor], info: Dict[str, torch.Tensor]
    ) -> Dict[str, list]:
        pred = info["l7"].detach().cpu().numpy()
        gt = batch["depth_map"].numpy()
        nyu = False
        num_samples = gt.shape[0]
        rms = np.zeros(num_samples, np.float32)
        log_rms = np.zeros(num_samples, np.float32)
        abs_rel = np.zeros(num_samples, np.float32)
        sq_rel = np.zeros(num_samples, np.float32)
        d1_all = np.zeros(num_samples, np.float32)
        a1 = np.zeros(num_samples, np.float32)
        a2 = np.zeros(num_samples, np.float32)
        a3 = np.zeros(num_samples, np.float32)

        for i in range(num_samples):
            gt_depth = gt[i][0]
            pred_depth = pred[i][0]
            mask = np.logical_and(gt_depth > self.min_depth, gt_depth < self.max_depth)
            
            if not nyu:
                gt_height, gt_width = gt_depth.shape
                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

            gt_depth = gt_depth[mask]            
            pred_depth = pred_depth[mask]
            scale = np.median(gt_depth) / np.median(pred_depth)
            pred_depth *= scale

            gt_depth, pred_depth, mask = self.process_depth(
                gt_depth, pred_depth, self.min_depth, self.max_depth)
            abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[
                i] = self.compute_errors(gt_depth, pred_depth, nyu=nyu)


        return {"abs_real":[abs_rel.mean()], "sq_rel":[sq_rel.mean()], "rms":[rms.mean()], "log_rms":[log_rms.mean()], "a1":[a1.mean()], "a2":[a2.mean()], "a3":[a3.mean()]}

    def _get_report(self) -> str:

        abs_real = np.array(self.abs_real).mean()
        sq_rel = np.array(self.sq_rel).mean()
        rms = np.array(self.rms).mean()
        log_rms = np.array(self.log_rms).mean()
        a1 = np.array(self.a1).mean()
        a2 = np.array(self.a2).mean()
        a3 = np.array(self.a3).mean()
        report = f"abs_real: {abs_real} \n sq_rel: {sq_rel} \n rms: {rms} \n log_rms:{log_rms} \n a1:{a1} \n a2:{a2} \n a3:{a3}"
        if self.save_to_disk:
            with open(self.report_path, "w") as handler:
                handler.write(report)
        return report

    def output(self, results: List[Dict[str, list]]):
        abs_real = reduce(lambda i, r: i + r["abs_real"], results, [])
        sq_rel = reduce(lambda i, r: i + r["sq_rel"], results, [])
        rms = reduce(lambda i, r: i + r["rms"], results, [])
        log_rms = reduce(lambda i, r: i + r["log_rms"], results, [])
        a1 = reduce(lambda i, r: i + r["a1"], results, [])
        a2 = reduce(lambda i, r: i + r["a2"], results, [])
        a3 = reduce(lambda i, r: i + r["a3"], results, [])
        self.abs_real = abs_real
        self.sq_rel = sq_rel
        self.rms = rms
        self.log_rms = log_rms
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self._get_report()
