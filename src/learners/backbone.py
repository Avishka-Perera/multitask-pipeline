from ..models.encoder import VisionTransformerBase, ResNet50, ResNet18
import torch
from typing import Dict, Sequence
from ..util import Logger
from omegaconf import OmegaConf
from ._base import BaseLearner
import numpy as np


class BackBoneLearner(BaseLearner):
    device_count = 1
    enc_Ds = {"ResNet50": 2048, "ResNet18": 512, "ViT-B": 768}
    valid_encs = enc_Ds.keys()

    def _init_encoder(self, enc_params: OmegaConf):
        if self.enc_name == "ResNet50":
            self.encoder = ResNet50(**dict(enc_params))
        elif self.enc_name == "ResNet18":
            self.encoder = ResNet18(**dict(enc_params))
        elif self.enc_name == "ViT-B":
            self.encoder = VisionTransformerBase(**dict(enc_params))
        self.emb_D = self.enc_Ds[self.enc_name]
        self.encoder.cuda(self.devices[0])

    def __init__(
        self,
        devices: Sequence[int],
        logger: Logger,
        enc_name: str,
        enc_params: Dict = {},
    ) -> None:
        super().__init__(devices=devices, logger=logger)
        if enc_name not in self.valid_encs:
            raise ValueError(
                f"Unsupported 'enc_name' definition. Supported encoders are {self.valid_encs}"
            )
        self.logger = logger
        self.devices = devices
        self.enc_name = enc_name
        self._init_encoder(enc_params)

    def forward(self, batch: Dict[str, torch.Tensor | int]) -> Dict[str, torch.Tensor]:
        imgs = batch["img"]

        *rest_axs, C, H, W = imgs.shape
        stacked_B = np.prod(rest_axs)
        with torch.no_grad():
            stacked_imgs = imgs.view(stacked_B, C, H, W)

        stacked_imgs = stacked_imgs.cuda(self.devices[0])
        out = self.encoder(stacked_imgs)
        for k, v in out.items():
            out[k] = v.view(([*rest_axs, -1]))

        return out
