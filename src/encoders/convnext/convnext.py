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
        depths=[None, None, 3, 3, 9, 3],
        dims=[96, 96, 96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()

        self.features = nn.ModuleList()  # 9 layers stem1-stem2-C-D-C-D-C-D-C
        stem1 = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=2, padding=1),
            nn.Dropout(0.1),
            LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
        )
        stem2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.1),
            LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
        )
        self.features.append(stem1)
        self.features.append(stem2)

        downsample_layer1 = nn.Sequential(
            LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2),
            nn.Dropout(0.1),
        )
        self.features.append(downsample_layer1)

        dp_rates = [
            x.item()
            for x in torch.linspace(
                0, drop_path_rate, sum([d for d in depths if d is not None])
            )
        ]
        cur = 0
        for i in range(2, 6):
            if i != 2:
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i - 1], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i - 1], dims[i], kernel_size=2, stride=2),
                    nn.Dropout(0.1),
                )
                self.features.append(downsample_layer)
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.features.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.dims = dims

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
            "f1": tenOne,
            "f2": tenTwo,
            "f3": tenThr,
            "f4": tenFou,
            "f5": tenFiv,
            "f6": tenSix,
        }
