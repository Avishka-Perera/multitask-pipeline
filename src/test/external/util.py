import torch
from ...util import is_lists_equal


def make_random_nested_tens(conf):
    if conf is None:
        return None
    elif type(conf) in [tuple, list] and all([type(v) == int for v in conf]):
        return torch.Tensor(*conf)
    elif type(conf) in [tuple, list]:
        nested_tens = []
        for sub_conf in conf:
            nested_tens.append(make_random_nested_tens(sub_conf))
        return nested_tens
    elif type(conf) == dict:
        nested_tens = {}
        for k, sub_conf in conf.items():
            nested_tens[k] = make_random_nested_tens(sub_conf)
        return nested_tens
    else:
        raise ValueError("Invalid Tensor configuration")


def are_shapes_equal(pack_1, pack_2):
    if type(pack_1) == type(pack_2) == torch.Tensor:
        if pack_1.shape != pack_2.shape:
            return False
        else:
            return True
    elif pack_1 is None and pack_2 is None:
        return True
    elif type(pack_1) == dict and type(pack_2) == dict:
        if is_lists_equal(list(pack_1.keys()), list(pack_2.keys())):
            return all([are_shapes_equal(pack_1[k], pack_2[k]) for k in pack_1.keys()])
        else:
            return False
    elif type(pack_1) in [list, tuple] and type(pack_2) in [list, tuple]:
        return all([are_shapes_equal(pack_1[i], pack_2[i]) for i in range(len(pack_1))])
    else:
        return False
