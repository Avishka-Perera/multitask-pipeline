from typing import Callable, Any, Tuple
import torch
from torch.nn import Sequential, Conv2d
from timm.models.convnext import ConvNeXt as timmConvNeXt
from timm.layers import LayerNorm2d


class ConvNeXt_T(timmConvNeXt):
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

        delattr(self, "head")
        delattr(self, "norm_pre")
        self.stem = Sequential(
            Conv2d(3, 96, kernel_size=(3, 3), stride=(2, 2), padding=1),
            Conv2d(96, 96, kernel_size=(4, 4), stride=(2, 2), padding=1),
            LayerNorm2d((96,), eps=1e-06),
        )
        self.stages[0].downsample = Sequential(  # TODO: is it safe to do this
            LayerNorm2d((96,), eps=1e-06),
            Conv2d(96, 96, kernel_size=(2, 2), stride=(2, 2)),
        )

        self.dims = {
            "f7": 3,
            "f6": 96,
            "f5": 96,
            "f4": 96,
            "f3": 192,
            "f2": 384,
            "f1": 768,
        }
        self.pyramid_level_names = ["f7", "f6", "f5", "f4", "f3", "f2", "f1"]

    def forward(self, x) -> torch.Tensor:
        f6 = self.stem[0](x)
        f5 = self.stem[1](f6)
        inter = self.stem[2](f5)
        f4 = self.stages[0](inter)
        f3 = self.stages[1](f4)
        f2 = self.stages[2](f3)
        f1 = self.stages[3](f2)
        emb = torch.mean(f1, dim=(-2, -1))
        return {
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f4": f4,
            "f5": f5,
            "f6": f6,
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
