import torch


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
