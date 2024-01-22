from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer
import os
from torchvision.datasets.utils import download_url
from ..constants import model_weights_dir
import logging

logger = logging.getLogger()


# based on https://github.com/facebookresearch/mae
class VisionTransformerBase(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    weights = {
        "MAE": {
            "url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
            "md5": "8cad7c8458a98d92977cf31e98c74644",
        }
    }
    embed_dim = 768

    def __init__(self, weights: str = None, global_pool=False, **kwargs):
        raise NotImplementedError("TODO: Expose dims and pyramid_levels as in ConvNeXt")
        super(VisionTransformerBase, self).__init__(
            patch_size=16,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
        )

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(self.embed_dim)

            del self.norm  # remove the original norm

        self.head = nn.Identity()
        if weights is not None:
            self.load_weights(weights)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    def _download_weights(self, name: str, path: str):
        logger.info(f"Downloading pretrained weights for {name}")

        root, filename = os.path.split(path)
        downloaded = False
        max_attempts = 2
        attempts = 1
        while not downloaded:
            try:
                download_url(
                    self.weights[name]["url"],
                    root,
                    filename,
                    self.weights[name]["md5"] if "md5" in self.weights[name] else None,
                )
                downloaded = True
            except RuntimeError as e:
                if str(e) != "File not found or corrupted.":
                    raise e
                elif attempts >= max_attempts:
                    raise RuntimeError(
                        f"Possible invalid checksum definition. Failed {attempts} attempts."
                    )
                else:
                    attempts += 1
                    logger.warn(
                        f"Initial checksum failed. Downloading again (attempt={attempts})..."
                    )

    def load_weights(self, weights: str):
        valid_weights = list(self.weights.keys())
        if weights not in valid_weights:
            raise ValueError(
                f"Invalid weights definition. Valid weights are {valid_weights}"
            )

        weights_path = os.path.join(
            model_weights_dir, "encoders", "vit-b", f"{weights.lower()}.pth"
        )
        if not os.path.exists(weights_path):
            self._download_weights(weights, weights_path)
        sd = torch.load(weights_path)
        self.load_state_dict(sd["model"])
