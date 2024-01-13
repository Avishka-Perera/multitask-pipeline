import sys
import os

root_dir = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(root_dir)
sys.path.remove(os.path.dirname(__file__))

import os
from argparse import ArgumentParser
import glob
from mt_pipe.src.test.external.datasets import test as test_datasets
from mt_pipe.src.test.external.learners import test as test_learners
from mt_pipe.src.test.external.models import test as test_models
from mt_pipe.src.util import Logger
import yaml
from omegaconf import OmegaConf


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="all",
        help="Defines what to test. Options: all, flow, datasets",
        choices=[
            "all",
            "datasets",
            "models",
            "learners",
            "losses",
            "evaluators",
            "other",
        ],
    )
    parser.add_argument(
        "-c",
        "--conf-dir",
        type=str,
        default="test",
        help="Defines what devices to use for testing",
    )
    parser.add_argument(
        "-d",
        "--devices",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Defines what devices to use for testing",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets",
        help="The data root",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="The directory containing pretrained model weights",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default="temp/test-out",
        help="The directory where test outputs will be saved",
    )
    parser.add_argument(
        "--dataset-test-cnt",
        type=int,
        default=6,
        help="Number of samples to randomly select from a dataset",
    )
    parser.add_argument(
        "-r",
        "--replica-size",
        type=int,
        default=2,
        help="Number of CUDA devices to train a single replica on",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.conf_dir):
        raise FileNotFoundError(
            f"The provided configs directory ({args.conf_dir}) dose not exist."
        )

    logger = Logger(1)

    conf_paths = glob.glob(f"{args.conf_dir}/*.yaml")
    conf_names = [".".join(os.path.split(p)[1].split(".")[:-1]) for p in conf_paths]
    valid_conf_names = ["datasets", "learners", "models"]
    assert all(
        [n in valid_conf_names for n in conf_names]
    ), f"Invalid configuration file name"

    for conf_path, conf_name in zip(conf_paths, conf_names):
        with open(conf_path) as handler:
            conf = OmegaConf.create(yaml.load(handler, yaml.FullLoader))

        if args.mode in ["all", conf_name]:
            if conf_name == "datasets":
                test_datasets(logger=logger, conf=conf, test_cnt=args.dataset_test_cnt)
                logger.info()
            if conf_name == "learners":
                test_learners(logger=logger, conf=conf, devices=args.devices)
                logger.info()
            if conf_name == "models":
                test_models(logger=logger, conf=conf)
                logger.info()
