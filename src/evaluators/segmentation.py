import os
import torch
import numpy as np
from typing import Dict, List
from mt_pipe.src.evaluators import BaseEvaluator
from matplotlib import cm, colors
from skimage import measure
from PIL import Image
import numpy as np

class SegmentationEvaluator(BaseEvaluator):
    def __init__(self, out_path: str = None, batch_img_key: str = "img") -> None:
        if out_path is not None:
            self.set_out_path(out_path)
        else:
            self.save_to_disk = False
            self.vis_path = None
            self.conf_mat_path = None
            self.report_path = None
        self.seg_id = 1
        self.batch_img_key = batch_img_key

    def set_out_path(self, out_path: str) -> None:
        self.save_to_disk = True
        self.vis_path = os.path.join(out_path, "visualizations")
        os.makedirs(self.vis_path, exist_ok=True)
        self.report_path = os.path.join(out_path, "report.txt")

    def _gray2rgb(self, gray_img):
        norm = colors.Normalize(vmin=gray_img.min(), vmax=gray_img.max())
        rgba_image = cm.gray(norm(gray_img))
        rgb_image = rgba_image[:, :, :3]
        return rgb_image

    def process_batch(
        self, batch: Dict[str, torch.Tensor], info: Dict[str, torch.Tensor]
    ) -> None:
        logits = info["logits"].cpu().detach().numpy() #80 128 128
        # logits = logits.argmax(axis=1)
        labels = batch["seg"].cpu().detach().numpy() #128 128
        images = batch[self.batch_img_key].cpu().detach().numpy()
        labels = labels.squeeze()
        total_intersec, total_union = 0, 0

        # labels_mask = np.zeros(labels.shape)
        # labels_mask[labels > 0.5 ] = 1
        # logits_mask = np.zeros(logits.shape)
        # logits_mask[logits > 0.5] = 1
        # intersection_mask = np.logical_and(labels_mask, logits_mask)
        # union_mask = np.logical_or(labels_mask, logits_mask)
        # intersect = intersection_mask.sum()
        # union = union_mask.sum()
        # total_intersec += intersect
        # total_union += union

        for j in range(logits.shape[0]):
            logits_mask = np.zeros(logits[j].shape)
            logits_mask[logits[j] > 0.5] = 1
            labels_ = labels[j]
            for i in np.unique(labels_):
                i = int(i)
                mask_1 = np.zeros_like(labels_)
                mask_1[labels_==i] = 1
                intersection_mask = np.logical_and(logits_mask[i], mask_1)
                union_mask = np.logical_or(logits_mask[i], mask_1)

                intersect = intersection_mask.sum()
                union = union_mask.sum()
                total_intersec += intersect
                total_union += union


        imgs = []
        for i in range(len(images)):
            img = self._visualize_segmentation(
                images[i], labels[i], logits[i], self.seg_id
            )
            imgs.append(img)

        return {"intersec": total_intersec, "union": total_union, "imgs": imgs}
    
    # def __visualize_segmentation(self, img, seg_lbl, seg_prd , seg_id):  #seg_lbl 128, 128   img 3, 128, 128 #seg_prd 80 128 128
    #     import matplotlib.pyplot as plt
    #     class_indices = np.unique(seg_prd)
    #     cmap = plt.cm.get_cmap('tab20', len(class_indices))
    #     color_map = cmap(np.arange(len(class_indices)))

    #     rgb_image = np.zeros((*seg_prd.shape, 3))
    #     for i, class_index in enumerate(class_indices):
    #         if class_index != 0:
    #             rgb_image[seg_prd == class_index] = color_map[i][:3] 
    #     conc = np.concatenate((img.transpose(1,2,0), rgb_image), 1)
    #     return conc 

    # def __visualize_segmentation(self, img, seg_prd, seg_lbl, seg_id): #seg_lbl 128, 128   img 3, 128, 128
    #     gt_binary_mask = np.zeros_like(seg_lbl)
    #     gt_binary_mask[seg_lbl == seg_id] = 1

    #     # seg_prd = seg_prd
    #     pd_binary_mask = np.zeros_like(seg_prd)
    #     pd_binary_mask[seg_prd == seg_id] = 1

    #     if img.shape[-1] == 1:
    #         img = self._gray2rgb(img.squeeze())

    #     # predicted regions
    #     transparency = 1
    #     for i in [1, 2]:
    #         img[i, :, :][pd_binary_mask == 1] = (
    #             img[i, :, :][pd_binary_mask > 0] * (1 - transparency)
    #             + pd_binary_mask[pd_binary_mask > 0] * transparency
    #         )
    #     for i in [0]:
    #         img[i, :, :][pd_binary_mask == 1] = img[i, :, :][pd_binary_mask > 0] * (
    #             1 - transparency
    #         )

    #     # ground truth borders
    #     contours = measure.find_contours(gt_binary_mask, 0.5)
    #     for contour in contours:
    #         contour = np.round(contour).astype(int)
    #         img[contour[:, 0], contour[:, 1], 2] = 0
    #         img[contour[:, 0], contour[:, 1], 0:2] = 1
    #     img = (img * 255).astype(np.uint8)

    #     return img
    
    def _visualize_segmentation(self, img, seg_lbl, seg_prd, seg_id):
        seg_prd = np.asarray(seg_prd)
        colormap = np.random.rand(80, 3)
        masked_clormap = np.ones((*seg_prd.shape, 3)) * colormap[:, None, None] * seg_prd[:,:,:,None]  #128,128,3 x 3 x 
        img1 = masked_clormap.sum(axis=(0))
        conc = np.concatenate([img.transpose(1, 2, 0), img1], 1)
        return conc

    def _save_report(self, iou: float) -> str:
        report = f"IoU: {iou}"
        if self.save_to_disk:
            with open(self.report_path, "w") as handler:
                handler.write(report)
        return report

    def _export_imgs(self, imgs):
        for i, img in enumerate(imgs):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            img.save(os.path.join(self.vis_path, f"{i}.jpg"))

    def output(self, results: List[Dict[str, int]]) -> str:
        ious = [res["intersec"] / res["union"] for res in results]
        iou = sum(ious) / len(ious)
        imgs = [img for res in results for img in res["imgs"]]
        if self.save_to_disk:
            self._export_imgs(imgs)
        return self._save_report(iou)
