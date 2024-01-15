import torch
from torch import nn, Tensor
from typing import Dict
from timm.models.layers import trunc_normal_
from .blocks import ConvNeXtBlock, LayerNorm


class ConvNeXt(nn.Module):
    r"""ConvNeXtEncoder
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        depths={"f7": None, "f6": None, "f5": None, "f4": 3, "f3": 3, "f2": 9, "f1": 3},
        dims={"f7": 3, "f6": 96, "f5": 96, "f4": 96, "f3": 192, "f2": 384, "f1": 768},
        pyramid_level_names=["f7", "f6", "f5", "f4", "f3", "f2", "f1"],
        drop_path_rate=0.5,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()

        self.dims = dims
        self.pyramid_level_names = pyramid_level_names

        self.features = nn.ModuleList()  # 9 layers stem1-stem2-C-D-C-D-C-D-C
        stem1 = nn.Sequential(
            nn.Conv2d(
                in_chans,
                dims[self.pyramid_level_names[1]],
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Dropout(0.1),
            LayerNorm(
                dims[self.pyramid_level_names[2]],
                eps=1e-6,
                data_format="channels_first",
            ),
        )
        stem2 = nn.Sequential(
            nn.Conv2d(
                dims[self.pyramid_level_names[1]],
                dims[self.pyramid_level_names[2]],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.Dropout(0.1),
            LayerNorm(
                dims[self.pyramid_level_names[2]],
                eps=1e-6,
                data_format="channels_first",
            ),
        )
        self.features.append(stem1)
        self.features.append(stem2)

        downsample_layer1 = nn.Sequential(
            LayerNorm(
                dims[self.pyramid_level_names[2]],
                eps=1e-6,
                data_format="channels_first",
            ),
            nn.Conv2d(
                dims[self.pyramid_level_names[2]],
                dims[self.pyramid_level_names[3]],
                kernel_size=2,
                stride=2,
            ),
            nn.Dropout(0.1),
        )
        self.features.append(downsample_layer1)

        depths_lst = list(depths.values())
        dp_rates = torch.linspace(
            0, drop_path_rate, sum([d for d in depths_lst if d is not None])
        ).tolist()
        dp_rates = {
            k: dp_rates[
                sum([d for d in depths_lst[:i] if d is not None]) : sum(
                    [d for d in depths_lst[:i] if d is not None]
                )
                + v
            ]
            for i, (k, v) in enumerate(depths.items())
            if v is not None
        }

        for i in range(3, len(pyramid_level_names)):
            l = pyramid_level_names[i]
            l_before = pyramid_level_names[i - 1]
            if l != self.pyramid_level_names[3]:
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[l_before], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[l_before], dims[l], kernel_size=2, stride=2),
                    nn.Dropout(0.1),
                )
                self.features.append(downsample_layer)
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=dims[l],
                        drop_path=dp_rates[l][j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[l])
                ]
            )
            self.features.append(stage)

        self.norm = nn.LayerNorm(
            dims[self.pyramid_level_names[-1]], eps=1e-6
        )  # final norm layer

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, tenInput: Tensor) -> Dict[str, Tensor]:
        tenSix = self.features[0](tenInput)  # stem1
        tenFiv = self.features[1](tenSix)  # stem2
        tenFiv1 = self.features[2](tenFiv)
        tenFou = self.features[3](tenFiv1)  # layer1
        tenFou1 = self.features[4](tenFou)
        tenThr = self.features[5](tenFou1)  # layer2
        tenThr1 = self.features[6](tenThr)
        tenTwo = self.features[7](tenThr1)  # layer3
        tenTwo1 = self.features[8](tenTwo)
        tenOne = self.features[9](tenTwo1)  # layer4

        emb = self.norm(
            tenOne.mean([-2, -1])
        )  # global average pooling, (N, C, H, W) -> (N, C)

        return {
            "emb": emb,
            self.pyramid_level_names[-1]: tenOne,
            self.pyramid_level_names[5]: tenTwo,
            self.pyramid_level_names[4]: tenThr,
            self.pyramid_level_names[3]: tenFou,
            self.pyramid_level_names[2]: tenFiv,
            self.pyramid_level_names[1]: tenSix,
        }
