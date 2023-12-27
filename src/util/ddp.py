import os
import torch.distributed as dist
import torch.multiprocessing as mp
import socket


def is_port_available(port: int) -> bool:
    """
    Check if a given port is available.
    :param port: Port to check
    :return: True if the port is available, False otherwise
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)

    try:
        sock.bind(("localhost", port))
        return True
    except socket.error:
        return False
    finally:
        sock.close()


def setup(rank, world_size):
    port = 12355
    while not is_port_available(port):
        port += 1

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def spawn(func, world_size, args):
    mp.spawn(func, args=(world_size, args), nprocs=world_size, join=True)
