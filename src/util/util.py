from torch.nn import Module
import requests
from PIL import Image
from io import BytesIO
import os
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import re
import yaml
import importlib
import omegaconf
from omegaconf.listconfig import ListConfig
from collections import OrderedDict
import random
import boto3
import logging

logger = logging.getLogger()


def are_lists_equal(lst1, lst2):
    new_lst2 = [*lst2]
    try:
        for obj in lst1:
            new_lst2.remove(obj)
    except ValueError:
        return False

    return len(new_lst2) == 0


def is_super_list(list1, list2):
    return set(list1) >= set(list2)


def open_network_img(url: str) -> Image:
    res = requests.get(url)
    if res.status_code == 200:
        img = Image.open(BytesIO(res.content))
        return img
    else:
        raise RuntimeError("404: Image was not found")


def images_to_gif(input_dir: str, output_path: str, duration=100, quality=75):
    """
    Convert a set of images in a directory to an animated GIF.

    :param input_dir: The directory containing the input images.
    :param output_gif: The path to the output animated GIF file.
    :param duration: The duration (in milliseconds) for each frame (default is 100 ms).
    """
    image_files = sorted(
        [f for f in os.listdir(input_dir) if f.endswith(".png") or f.endswith(".jpg")]
    )

    images = []
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        img = Image.open(image_path)
        images.append(img)

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=duration,
        quality=quality,
    )


def load_class(target):
    """loads a class using a target"""
    *module_name, class_name = target.split(".")
    module_name = ".".join(module_name)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


def make_obj_from_conf(conf, **kwargs):
    cls = load_class(conf["target"])
    params = conf["params"] if "params" in conf else {}
    obj = cls(**params, **kwargs)
    return obj


# loads scientific notations as float
yaml_loader = yaml.SafeLoader
yaml_loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def load_config(path):
    with open(path) as handler:
        config = yaml.load(handler, yaml_loader)
    config = omegaconf.OmegaConf.create(config)
    return config


# credits: https://gist.github.com/gatheluck/c57e2a40e3122028ceaecc3cb0d152ac
def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_nested_attribute(obj, keys):
    """
    Retrieve a nested attribute from an object using a string of keys.

    Args:
        obj: The object to search within.
        keys: A string containing keys separated by dots.

    Returns:
        The value of the nested attribute or None if not found.
    """
    keys = keys.strip(".")
    if keys == "":
        return obj

    key_list = keys.split(".")
    current_obj = obj

    for key in key_list:
        current_obj = getattr(current_obj, key)

    return current_obj


def has_all_vals(target, values) -> bool:
    return all([val in target for val in values])


def has_excess_vals(target, values) -> bool:
    return any([tar not in values for tar in target])


def validate_keys(target, required_vals, possible_vals=None, name=None) -> None:
    assert name is not None
    if not has_all_vals(target, required_vals):
        raise AttributeError(f"The keys {required_vals} are required for '{name}'")
    if possible_vals is not None:
        if has_excess_vals(target, possible_vals):
            raise AttributeError(
                f"Unexpected key found for '{name}'. Possible keys are {possible_vals}"
            )


def fix_list_len(lst, exp_len):
    """
    Fixes the length of a given list to desired length either;
        1. by repeating (e.g.: [a, b, c] -> [a, b, c, a, b, c, a, b] when expected length is 8)
        2. by slicing
    """
    if len(lst) == exp_len:
        return lst
    elif len(lst) > exp_len:
        return lst[:exp_len]
    else:
        rep_cnt = exp_len // len(lst)
        rem_cnt = exp_len % len(lst)
        return lst * rep_cnt + lst[:rem_cnt]


def split_s3_uri(s3_uri):
    trimmed = s3_uri[5:]
    bucket_name, *key = trimmed.split("/")
    key = "/".join(key)
    return bucket_name, key


def download_with_pbar(s3, bucket_name, object_key, local_file_path):
    total_size = s3.head_object(Bucket=bucket_name, Key=object_key)["ContentLength"]
    with tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"Downloading: {object_key} to {local_file_path}",
    ) as pbar:
        with open(local_file_path, "wb") as f:
            s3.download_fileobj(
                bucket_name,
                object_key,
                f,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )


def download_s3_directory(s3_uri, local_path, show_pbar=False):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    bucket_name, prefix = split_s3_uri(s3_uri)
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    for page in pages:
        for content in page.get("Contents", []):
            object_key = content["Key"]
            local_file_path = os.path.join(
                local_path, object_key.replace(prefix, "").strip("/")
            )
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            if show_pbar:
                download_with_pbar(s3, bucket_name, object_key, local_file_path)
            else:
                s3.download_file(bucket_name, object_key, local_file_path)
                logger.info(f"Downloaded: {object_key} to {local_file_path}")


def load_model_states(
    model: Module, state_dict: OrderedDict, model_map_info: ListConfig
) -> None:
    for i, layer_map_info in enumerate(model_map_info):
        source = layer_map_info.source
        target = layer_map_info.target
        module = get_nested_attribute(model, target)
        if source == ".":
            status = module.load_state_dict(
                state_dict,
            )
            if logger is not None:
                logger.info(f"{i}, {status}")
        else:
            source = source.strip(".")
            new_sd = {
                k.lstrip(source).strip("."): v
                for (k, v) in state_dict.items()
                if k.startswith(source)
            }
            status = module.load_state_dict(new_sd)
            if logger is not None:
                logger.info(f"{i}, {status}")


def flatten_leads(tens: torch.Tensor, dim_count: int) -> torch.Tensor:
    merge_dims = tens.shape[:dim_count]
    unchn_dims = tens.shape[dim_count:]
    new_shape = [np.prod(merge_dims), *unchn_dims]
    return tens.view(*new_shape)


def has_inner_dicts(dictionary):
    for value in dictionary.values():
        if isinstance(value, dict):
            return True
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, dict):
                    return True

    return False


def get_shallow_vals(dic):
    shallow_dic = {}
    for i in dic.keys():
        if i != "tot":
            if type(dic[i]) == dict:
                if dic[i]["tot"] is not None:
                    shallow_dic[i] = dic[i]["tot"]
            else:
                if dic[i] is not None:
                    shallow_dic[i] = dic[i]
    return shallow_dic
