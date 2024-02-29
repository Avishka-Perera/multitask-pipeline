import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecision
from typing import Tuple
from looseversion import LooseVersion
from torch.cuda.amp import GradScaler


def get_is_dist() -> bool | Tuple[int, int, int]:
    if all([k in os.environ for k in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]]):
        return [int(os.environ[k]) for k in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]]
    else:
        return False


def setup():
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


# This is only supported in ampere architecture GPUs (3090, A10, A100...). If you use this in a non-amp arch, it won't throw any error, but the training will be significantly slower
bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,  # Param precision
    reduce_dtype=torch.bfloat16,  # Gradient communication precision
    buffer_dtype=torch.bfloat16,  # Buffer precision
)

# Available in all GPUs; much slower compared to bfloat16 (4% in A100), low dynamic range of values (can cause exploding gradients), must need to rescale the gradients using the grad scaler
fpSixteen = MixedPrecision(
    param_dtype=torch.float16,  # Param precision
    reduce_dtype=torch.float16,  # Gradient communication precision
    buffer_dtype=torch.float16,  # Buffer precision
)


def get_mixed_prec():
    bf16_ready = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and LooseVersion(torch.version.cuda) >= "11.0"
        and dist.is_nccl_available()
        and torch.cuda.nccl.version() >= (2, 10)
    )

    if bf16_ready:
        return torch.bfloat16, None
    else:
        return torch.float16, GradScaler()
