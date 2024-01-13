import torch
import numpy as np
from typing import Tuple
from ...util import are_lists_equal, load_class


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
