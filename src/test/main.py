from argparse import ArgumentParser
import logging
import random
import sys
import os

root_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
sys.path.append(root_dir)
from src.util import set_all_seeds, Logger
from src.test.datasets import test as test_dataset
from src.test.models import test as test_model
from src.test.losses import test as test_cost
from src.test.learners import test as test_learner
from src.test.other import test as test_other
from src.test.evaluators import test as test_evaluator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s:%(levelname)s: %(message)s"
)
seed = round(random.random() * 100)
print(f"Using seed: {seed}")
set_all_seeds(seed)


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
        "-c",
        "--conf-dir",
        type=str,
        default="src/configs/20",
        help="The directory containing trainer configurations",
    )
    parser.add_argument(
        "--dataset-test-cnt",
        type=int,
        default=5,
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
    logger = Logger(1)
    logger.info("==== Starting Testing ====")
    if args.mode == "all" or args.mode == "datasets":
        logger.info("")
        test_dataset(
            data_dir=args.data_dir, test_cnt=args.dataset_test_cnt, logger=logger
        )

    if args.mode == "all" or args.mode == "models":
        logger.info("")
        test_model(model_dir=args.model_dir, devices=args.devices, logger=logger)

    if args.mode == "all" or args.mode == "learners":
        logger.info("")
        test_learner(devices=args.devices, logger=logger)

    if args.mode == "all" or args.mode == "losses":
        logger.info("")
        test_cost(device=args.devices[0], logger=logger)

    if args.mode == "all" or args.mode == "evaluators":
        logger.info("")
        test_evaluator(out_dir=args.out_dir, logger=logger)

    if args.mode == "all" or args.mode == "other":
        logger.info("")
        test_other(data_dir=args.data_dir, logger=logger)
