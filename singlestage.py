import os
import sys

root_dir = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(root_dir)
if os.path.dirname(__file__) in sys.path:
    sys.path.remove(os.path.dirname(__file__))

import argparse
import ast
from mt_pipe.src.util import Trainer
from mt_pipe.src.util import set_all_seeds, Logger
import torch
from mt_pipe.src.util.ddp import setup, cleanup, spawn
from mt_pipe.src.constants import analysis_levels, log_levels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, help="path to the configuration file", required=True
    )
    parser.add_argument(
        "-d",
        "--devices",
        type=ast.literal_eval,
        nargs="+",
        help="List of device IDs",
        default=list(range(torch.cuda.device_count())),
    )
    parser.add_argument(
        "-r",
        "--replica-size",
        type=int,
        help="Number of devices to be used in a single replica",
        default=2,
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        help="The directory to resume training",
        default=None,
    )
    parser.add_argument(
        "--force-resume",
        help="Whether to resume the training job irrespective of the configuration",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Where outputs and logs are saved",
        default="out",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Trailing directory name after '--output-dir'. Use this causiously since this will overright any existing files",
        default=None,
    )
    parser.add_argument(
        "--mb",
        "--mock-batch-count",
        dest="mock_batch_count",
        type=ast.literal_eval,
        nargs="+",
        default=[-1],
        help="limits the number of batches used for fitting",
    )
    parser.add_argument(
        "--me",
        "--mock-epoch-count",
        dest="mock_epoch_count",
        type=int,
        default=-1,
        help="limits the number of epochs used for fitting",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=1,
        help="Logging level. 0: notset, 1: info, 2: warn, 3: error",
        choices=log_levels,
    )
    parser.add_argument(
        "-a",
        "--analysis-level",
        type=int,
        default=1,
        help="The level of analysis to do. 0: no analysis; 1: break loss into parts; 2: break loss into parts and analyze gradients",
        choices=analysis_levels,
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        help="Checkpoints file to load states from",
        default=None,
    )
    parser.add_argument(
        "--ckpt-map-conf-path",
        type=str,
        help="File describing the map of weights",
        default=None,
    )
    args = parser.parse_args()
    return args


def main(rank, world_size, args):
    logger = Logger(args.verbose, rank)
    logger.info(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    # setup mp_model and devices for the process
    devices = [
        args.devices[rank * args.replica_size + i] for i in range(args.replica_size)
    ]

    trainer = Trainer(
        args.config,
        weights_conf={
            "ckpt_path": args.ckpt_path,
            "ckpt_map_conf_path": args.ckpt_map_conf_path,
        },
        devices=devices,
        rank=rank,
        world_size=world_size,
        logger=logger,
        analysis_level=args.analysis_level,
    )

    trainer.fit(
        output_path=args.output_path,
        run_name=args.run_name,
        mock_batch_count=args.mock_batch_count,
        mock_epoch_count=args.mock_epoch_count,
        resume_dir=args.resume_dir,
        force_resume=args.force_resume,
    )

    cleanup()


if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        set_all_seeds(args.seed)

    replica_size = args.replica_size
    n_gpus = len(args.devices)
    assert (
        n_gpus >= replica_size
    ), f"Requires at least {replica_size} GPUs to run, but got {n_gpus}"
    world_size = n_gpus // replica_size
    if args.verbose != 0:
        print(f"replica_size: {replica_size}, world_size: {world_size}")
    spawn(main, world_size, args)
