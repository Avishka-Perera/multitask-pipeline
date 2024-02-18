from typing import Callable, Any, Tuple
import torch
from torch.nn import Sequential, Conv2d
from timm.models.convnext import ConvNeXt as timmConvNeXt
from timm.layers import LayerNorm2d


class ConvNeXt_T(timmConvNeXt):
    # required attributes for the mt_pipe architecture
    dims = {
        "l1": 48,
        "l2": 48,
        "l3": 96,
        "l4": 192,
        "l5": 384,
        "l6": 768,
    }

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "avg",
        output_stride: int = 32,
        depths: Tuple[int, ...] = ...,
        dims: Tuple[int, ...] = ...,
        kernel_sizes: int | Tuple[int, ...] = 7,
        ls_init_value: float | None = 0.000001,
        stem_type: str = "patch",
        patch_size: int = 4,
        head_init_scale: float = 1,
        head_norm_first: bool = False,
        head_hidden_size: int | None = None,
        conv_mlp: bool = False,
        conv_bias: bool = True,
        use_grn: bool = False,
        act_layer: str | Callable[..., Any] = "gelu",
        norm_layer: str | Callable[..., Any] | None = None,
        norm_eps: float | None = None,
        drop_rate: float = 0,
        drop_path_rate: float = 0,
    ):
        # configuration of ConvNeXt-T
        depths = (3, 3, 9, 3)
        dims = (96, 192, 384, 768)
        super().__init__(
            in_chans,
            num_classes,
            global_pool,
            output_stride,
            depths,
            dims,
            kernel_sizes,
            ls_init_value,
            stem_type,
            patch_size,
            head_init_scale,
            head_norm_first,
            head_hidden_size,
            conv_mlp,
            conv_bias,
            use_grn,
            act_layer,
            norm_layer,
            norm_eps,
            drop_rate,
            drop_path_rate,
        )

        # delete the unwanted head
        delattr(self, "head")
        delattr(self, "norm_pre")

        # replace the original stem layer ([Conv4x4 -> LayerNorm]) with a new stem layer ([Conv3x3 -> Conv4x4 -> LayerNorm])
        self.stem = Sequential(
            Sequential(
                Conv2d(3, 48, kernel_size=(4, 4), stride=(2, 2), padding=1),
                LayerNorm2d((48,), eps=1e-06),
            ),
            Sequential(
                Conv2d(48, 48, kernel_size=(3, 3), stride=(2, 2), padding=1),
                LayerNorm2d((48,), eps=1e-06),
            ),
        )

        # add an additional downsample at the first stage (originally Identity)
        self.stages[0].downsample = Sequential(  #
            LayerNorm2d((48,), eps=1e-06),
            Conv2d(48, 96, kernel_size=(2, 2), stride=(2, 2)),
        )

    def forward(self, x) -> torch.Tensor:
        f1 = self.stem[0](x)
        f2 = self.stem[1](f1)
        f3 = self.stages[0](f2)
        f4 = self.stages[1](f3)
        f5 = self.stages[2](f4)
        f6 = self.stages[3](f5)
        emb = torch.mean(f6, dim=(-2, -1))
        return {
            "l1": f1,
            "l2": f2,
            "l3": f3,
            "l4": f4,
            "l5": f5,
            "l6": f6,
            "emb": emb,
        }

    def forward_features(self, x):
        """Method inherited from parent class"""
        raise NotImplementedError("This method will not be implemented")

    def get_classifier(self):
        """Method inherited from parent class"""
        raise NotImplementedError("This method will not be implemented")

    def reset_classifier(self, num_classes=0, global_pool=None):
        """Method inherited from parent class"""
        raise NotImplementedError("This method will not be implemented")

    def forward_head(self, x, pre_logits: bool = False):
        """Method inherited from parent class"""
        raise NotImplementedError("This method will not be implemented")
