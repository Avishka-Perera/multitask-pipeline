import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecision
from typing import Tuple


def get_is_dist() -> bool | Tuple[int, int, int]:
    if all([k in os.environ for k in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]]):
        return [int(os.environ[k]) for k in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]]
    else:
        return False


def setup():
    # initialize the process group
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
