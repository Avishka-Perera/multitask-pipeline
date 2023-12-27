from ..util import Logger
from typing import Sequence
import subprocess
import glob


def test(conf_dir: str, replica_size: int, devices: Sequence[int], logger: Logger):
    logger.info("Testing Trainer...")
    confs = [
        p for p in glob.glob(f"{conf_dir}/**", recursive=True) if p.endswith(".yaml")
    ]
    for conf in confs:
        logger.info(f"Initiating {conf}...")
        command = [
            "python",
            "main.py",
            "--mock-batch-count",
            "2",
            "--mock-epoch-count",
            "2",
            "--devices",
            *[str(d) for d in devices],
            "-r",
            str(replica_size),
            "-c",
            conf,
            "-v",
            "0",
        ]
        subprocess.run(command, check=True)
