import torch
import numpy as np
from typing import Tuple
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from ...util import are_lists_equal, load_class


def make_random_nested_tens(conf):
    if type(conf) == DictConfig or type(conf) == ListConfig:
        conf = OmegaConf.to_container(conf)

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


def validate_nested_obj(obj, conf, tentative_none_mask=None) -> Tuple[bool, str]:
    def validate_nested_objs(
        obj, conf, key_lead="", tentative_none_mask=None
    ) -> Tuple[bool, str]:
        if type(obj) not in [tuple, list, dict]:  # termination condition
            if obj is None:
                if conf is None:
                    return True, "Valid"
                else:
                    if tentative_none_mask is not None:
                        try:
                            eval(f"tentative_none_mask{key_lead}")
                            return True, "Valid"
                        except KeyError as err:
                            if str(err).strip("'").strip('"') in key_lead:
                                return False, f"Unexpected object None. key: {key_lead}"
                            else:
                                raise err
                    else:
                        return False, f"Unexpected object None. key: {key_lead}"
            else:
                if type(obj) in [np.ndarray, torch.Tensor]:
                    if "shape" in conf:  #  shape is a required definition
                        if not are_lists_equal(list(conf["shape"]), list(obj.shape)):
                            return (
                                False,
                                f"Invalid shape. Key: {key_lead}. Expected: {conf['shape']}, Found: {list(obj.shape)}",
                            )

                        # other optional checks
                        if "min" in conf and conf["min"] > obj.min():
                            return (
                                False,
                                f"Invalid minimum. Key: {key_lead}. Expected: {conf['min']}, Found: {obj.min()}",
                            )
                        if "max" in conf and conf["max"] < obj.max():
                            return (
                                False,
                                f"Invalid maximum. Key: {key_lead}. Expected: {conf['max']}, Found: {obj.max()}",
                            )
                        if "dtype" in conf and load_class(conf["dtype"]) != obj.dtype:
                            return (
                                False,
                                f"Invalid dtype. Key: {key_lead}. Expected: {conf['dtype']}, Found: {obj.dtype}",
                            )
                        if "unique" in conf and not are_lists_equal(
                            list(conf.unique), list(obj.unique())
                        ):
                            return (
                                False,
                                f"Uniques dose not match. Key: {key_lead}. Expected: {conf['unique']}, Found: {obj.unique()}",
                            )

                        return True, "Valid"
                    else:
                        return (
                            False,
                            f"Unexpected object. Key: {key_lead}.",
                        )  # If the shape is not there, that means this is an unexpected obj

                elif type(obj) in [int, float]:
                    if "dtype" in conf:
                        if "min" in conf and conf["min"] > obj.min():
                            return (
                                False,
                                f"Invalid minimum. Key: {key_lead}. Expected: {conf['min']}, Found: {obj.min()}",
                            )
                        if "max" in conf and conf["max"] < obj.max():
                            return (
                                False,
                                f"Invalid maximum. Key: {key_lead}. Expected: {conf['max']}, Found: {obj.max()}",
                            )
                        return True, "Valid"
                    else:
                        return (
                            False,
                            f"Unexpected object. Key: {key_lead}.",
                        )  # If the dtype is not there, that means this is an unexpected obj
                else:
                    raise NotImplementedError(
                        "Not implemented for other objects yet. Please open an issue with your use case"
                    )

        elif type(obj) in [tuple, list]:
            if type(conf) not in [tuple, list, ListConfig]:
                return False, f"conf and obj types dose not match. Key: {key_lead}"
            if len(conf) != len(obj):
                return False, f"conf and obj have different sizes. Key: {key_lead}"

            for i, sub_obj in enumerate(obj):
                if i < len(conf):
                    valid, msg = validate_nested_objs(
                        sub_obj,
                        conf[i],
                        key_lead=f"{key_lead}[{i}]",
                        tentative_none_mask=tentative_none_mask,
                    )
                    if not valid:
                        return False, msg
                else:
                    return False, f"Unexpected object. Key: {key_lead}[{k}]"

        else:  # type(obj) = dict
            if type(conf) not in [dict, DictConfig]:
                return False, f"conf and obj types dose not match. Key: {key_lead}"
            if not are_lists_equal(list(conf.keys()), list(obj.keys())):
                return False, f"conf and obj have different keys. Key: {key_lead}"

            for k, sub_obj in obj.items():
                lead = f'{key_lead}["{k}"]' if type(k) == str else f"{key_lead}[{k}]"
                if k in conf:
                    valid, msg = validate_nested_objs(
                        sub_obj,
                        conf[k],
                        key_lead=lead,
                        tentative_none_mask=tentative_none_mask,
                    )
                    if not valid:
                        return False, msg
                else:
                    return False, f"Unexpected object. Key: {lead}"

        return True, "Valid"

    def validate_nested_conf_keys(obj, conf, key_lead="") -> Tuple[bool, str]:
        if type(obj) in [tuple, list]:
            if len(obj) == len(conf):
                for i, sub_obj in obj:
                    validate_nested_conf_keys(
                        sub_obj, conf[i], key_lead=f"{key_lead}[{i}]"
                    )
            else:
                return (
                    False,
                    f"obj length is smaller than expected. Key: {key_lead}[{i}]",
                )
        elif type(obj) == dict:
            if are_lists_equal(list(obj.keys()), list(conf.keys())):
                for k, sub_obj in obj.items():
                    lead = (
                        f'{key_lead}["{k}"]' if type(k) == str else f"{key_lead}[{k}]"
                    )
                    validate_nested_conf_keys(sub_obj, conf[k], key_lead=lead)
            else:
                missing_keys = list(set(conf.keys()) - set(obj.keys()))
                missing_keys = [
                    f'{key_lead}["{k}"]' if type(k) == str else f"{key_lead}[{k}]"
                    for k in missing_keys
                ]
                return False, f"Missing obj for {missing_keys}"

        return True, "Valid"

    valid, msg = validate_nested_objs(
        obj, conf, tentative_none_mask=tentative_none_mask
    )
    if not valid:
        return False, msg
    valid, msg = validate_nested_conf_keys(obj, conf)

    return valid, msg


def are_shapes_equal(pack_1, pack_2):
    if type(pack_1) == type(pack_2) == torch.Tensor:
        if pack_1.shape != pack_2.shape:
            return False
        else:
            return True
    elif pack_1 is None and pack_2 is None:
        return True
    elif type(pack_1) == dict and type(pack_2) == dict:
        if are_lists_equal(list(pack_1.keys()), list(pack_2.keys())):
            return all([are_shapes_equal(pack_1[k], pack_2[k]) for k in pack_1.keys()])
        else:
            return False
    elif type(pack_1) in [list, tuple] and type(pack_2) in [list, tuple]:
        return all([are_shapes_equal(pack_1[i], pack_2[i]) for i in range(len(pack_1))])
    elif pack_1 == pack_2:
        return True
    else:
        return False
