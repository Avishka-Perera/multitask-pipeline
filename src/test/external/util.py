import torch
import numpy as np
from typing import Tuple
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from ...util import are_lists_equal, is_super_list, load_class


def make_random_nested_obj(conf):
    if type(conf) == DictConfig or type(conf) == ListConfig:
        conf = OmegaConf.to_container(conf)
    if conf is None:
        return None
    elif type(conf) in [tuple, list] and all([type(v) == int for v in conf]):
        return torch.randn(*conf)
    elif type(conf) in [tuple, list]:
        nested_tens = []
        for sub_conf in conf:
            nested_tens.append(make_random_nested_obj(sub_conf))
        return nested_tens
    elif type(conf) == dict:
        if are_lists_equal(conf.keys(), ["type", "value"]) and conf["type"] in [
            "str",
            "int",
            "float",
            "list",
        ]:
            if conf["type"] in ["str", "int", "float"]:
                return conf["value"]
            elif conf["type"] == "list":
                return list(conf["value"])
            raise ValueError("Invalid Tensor configuration")
        elif (
            all([k in conf.keys() for k in ["type", "shape"]])
            and conf["type"] == "torch.Tensor"
        ):
            if "unique" in conf:
                tens = torch.from_numpy(np.random.choice(conf["unique"], conf["shape"]))
                return tens
            else:
                tens = torch.randn(*conf["shape"])
                if "min" in conf:
                    tens[tens < conf["min"]] = conf["min"]
                if "max" in conf:
                    tens[tens > conf["max"]] = conf["max"]
                return tens
        else:
            nested_tens = {}
            for k, sub_conf in conf.items():
                nested_tens[k] = make_random_nested_obj(sub_conf)
            return nested_tens
    else:
        raise ValueError("Invalid Tensor configuration")


def validate_nested_obj(obj, conf, tentative_none_mask=None) -> Tuple[bool, str]:
    conf = OmegaConf.create(conf)

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
                        if "unique" in conf:
                            if type(obj) == np.ndarray:
                                uniques = np.unique(obj)
                            else:
                                uniques = obj.unique()
                            if not are_lists_equal(list(conf.unique), uniques.tolist()):
                                return (
                                    False,
                                    f"Uniques dose not match. Key: {key_lead}. Expected: {conf['unique']}, Found: {uniques.tolist()}",
                                )
                        if "unique_range" in conf:
                            if type(obj) == np.ndarray:
                                uniques = np.unique(obj)
                            else:
                                uniques = obj.unique()
                            if not is_super_list(
                                list(range(*conf.unique_range)), uniques.tolist()
                            ):
                                return (
                                    False,
                                    f"Unique range dose not match. Key: {key_lead}. Expected: range{tuple(conf['unique_range'])}, Found: {uniques.tolist()}",
                                )

                        return True, "Valid"
                    else:
                        return (
                            False,
                            f"1 Unexpected object. Key: {key_lead}.",
                        )  # If the shape is not there, that means this is an unexpected obj

                elif type(obj) in [int, float]:
                    if "dtype" in conf:
                        if "min" in conf and (
                            (type(obj) in [int, float] and conf["min"] > obj)
                            or (
                                (
                                    type(obj) not in [int, float]
                                    and conf["min"] > obj.min()
                                )
                            )
                        ):
                            return (
                                False,
                                f"Invalid minimum. Key: {key_lead}. Expected: {conf['min']}, Found: {obj if type(obj) in [float, int] else obj.min()}",
                            )
                        if "max" in conf and (
                            (type(obj) in [int, float] and conf["max"] < obj)
                            or (
                                type(obj) not in [int, float]
                                and conf["max"] < obj.max()
                            )
                        ):
                            return (
                                False,
                                f"Invalid maximum. Key: {key_lead}. Expected: {conf['max']}, Found: {obj if type(obj) in [float, int] else obj.max()}",
                            )
                        return True, "Valid"
                    elif conf != type(obj).__name__:
                        return (
                            False,
                            f"Found {type(obj).__name__} object. Expected conf: {conf}. Key: {key_lead}.",
                        )
                elif type(obj) == str:
                    if conf != "str":
                        return (
                            False,
                            f"Found 'str' object. Expected conf: {conf}. Key: {key_lead}",
                        )
                else:
                    raise NotImplementedError(
                        f"Not implemented for ({type(obj)}) objects yet. Key: {key_lead}. Please open an issue with your use case"
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
                    return False, f"3 Unexpected object. Key: {key_lead}[{k}]"

        else:  # type(obj) = dict
            if type(conf) not in [dict, DictConfig]:
                return False, f"conf and obj types dose not match. Key: {key_lead}"
            if not are_lists_equal(list(conf.keys()), list(obj.keys())):
                return (
                    False,
                    f"conf and obj have different keys. Key: {key_lead}. conf keys: {conf.keys()}. obj keys: {obj.keys()}",
                )

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
                    return False, f"4 Unexpected object. Key: {lead}"

        return True, "Valid"

    def validate_nested_conf_keys(obj, conf, key_lead="") -> Tuple[bool, str]:
        if type(obj) in [tuple, list]:
            if len(obj) == len(conf):
                for i, sub_obj in enumerate(obj):
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


def get_nested_obj_conf(obj):
    if type(obj) == dict:
        conf = dict()
        for k, v in obj.items():
            conf[k] = get_nested_obj_conf(v)
    elif type(obj) in [tuple, list]:
        conf = []
        for sub_obj in obj:
            conf.append(get_nested_obj_conf(sub_obj))
    elif type(obj) in [str, float, int]:
        return str(type(obj).__name__)
    elif type(obj) == torch.Tensor:
        return {"shape": list(obj.shape), "dtype": obj.dtype}
    else:
        raise TypeError(f"Unexpected object type ({type(obj)})")

    return conf


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
