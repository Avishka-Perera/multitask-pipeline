from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer
import os
from torchvision.datasets.utils import download_url


# based on https://github.com/facebookresearch/mae
class VisionTransformerBase(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    mae_weights = {
        "url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
        "md5": "8cad7c",
    }
    embed_dim = 768

    def __init__(self, weights_path=None, global_pool=False, **kwargs):
        super(VisionTransformerBase, self).__init__(
            patch_size=16,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs
        )

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(self.embed_dim)

            del self.norm  # remove the original norm

        self.head = nn.Identity()
        if weights_path is not None:
            self.load_weights(weights_path)

    def forward_features(self, x):
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
        embs = super().forward(x)
        return {"embs": embs}

    def _download_weights(self, path):
        root, filename = os.path.split(path)
        download_url(
            self.mae_weights["url"],
            root,
            filename,
            self.mae_weights["md5"],
        )

    def load_weights(self, path):
        if not os.path.exists(path):
            self._download_weights(path)
        sd = torch.load(path)
        self.load_state_dict(sd["model"])
