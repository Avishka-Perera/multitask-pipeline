from torch import nn
from typing import Sequence, Dict
import torch
from .backbone import BackBoneLearner
from omegaconf import OmegaConf
from ..util import Logger
from ._base import BaseLearner
import numpy as np


class ClassifierLearner(BaseLearner):
    device_count = 2

    def _init_head(self, head_trail, use_batch_norm, drop_out):
        head_dims = [self.emb_D, *head_trail]
        head_linear_layers = [
            nn.Linear(head_dims[i], head_dims[i + 1]) for i in range(len(head_dims) - 1)
        ]
        if use_batch_norm:
            layer_stack = [
                head_linear_layers,
                [nn.BatchNorm1d(i) for i in head_dims[1:]],
                [nn.ReLU()] * (len(head_linear_layers) - 1),
                [nn.Dropout(drop_out)] * (len(head_linear_layers) - 1),
            ]
        else:
            layer_stack = [
                head_linear_layers,
                [nn.ReLU()] * (len(head_linear_layers) - 1),
                [nn.Dropout(drop_out)] * (len(head_linear_layers) - 1),
            ]
        head_layers = [layer for layer_set in zip(*layer_stack) for layer in layer_set]
        head_layers.extend([head_linear_layers[-1], nn.Dropout(drop_out)])
        self.decoder = nn.Sequential(*head_layers)
        self.decoder.cuda(self.devices[1])
        self.out_D = head_trail[-1]

    def __init__(
        self,
        devices: Sequence[int],
        logger: Logger,
        enc_name: str,
        head_trail: Sequence[int],
        dec_params: OmegaConf = {},
        enc_params: OmegaConf = {},
    ) -> None:
        super().__init__(devices=devices, logger=logger)
        self.logger = logger
        self.enc_name = enc_name
        self.devices = devices

        self.encoder = BackBoneLearner(
            devices=devices, logger=logger, enc_name=enc_name, enc_params=enc_params
        )
        self.emb_D = self.encoder.emb_D
        self._init_head(head_trail, **dict(dec_params))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = self.encoder(batch)
        embs = out["embs"].cuda(self.devices[1])
        *rest_axs, D = embs.shape
        stacked_B = np.prod(rest_axs)
        stacked_embs = embs.view(stacked_B, D)
        stacked_logits = self.decoder(stacked_embs)

        logits = stacked_logits.view(*rest_axs, self.out_D)

        return {"embs": embs, "logits": logits}

    def load_weights(self, weights_path: str) -> None:
        conf = torch.load(weights_path)
        sd = conf["state"]["model"]
        self.load_state_dict(sd)
