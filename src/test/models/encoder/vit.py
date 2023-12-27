import os
from ....models import VisionTransformerBase
from ....util import Logger
from ....constants import img_wh
import torch
from ....util import Logger, is_lists_equal


def test(
    model_dir: str,
    device: int,
    logger: Logger,
) -> None:
    logger.info("Testing ViT-Base...")
    model = VisionTransformerBase(
        os.path.join(model_dir, "mae_pretrain_vit_base.pth")
    ).cuda(device)

    B, C, H, W = 8, 3, *img_wh[::-1]
    mock_imgs = torch.Tensor(B, C, H, W).cuda(device)
    out = model(mock_imgs)
    assert is_lists_equal(list(out.keys()), ["embs"])
    embs = out["embs"]
    assert embs.shape == torch.Size([B, model.embed_dim])
    assert embs.dtype == torch.float
