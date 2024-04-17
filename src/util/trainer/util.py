from typing import Sequence
from omegaconf import OmegaConf, ListConfig
from ..util import validate_keys, make_obj_from_conf
from ...datasets import BaseDataset
from ..logger import Logger


def load_datasets(conf) -> Sequence[BaseDataset] | BaseDataset:
    datasets = {}
    for ds_nm, ds_conf in conf.datasets.items():
        datasets[ds_nm] = make_obj_from_conf(ds_conf)
    return datasets


def validate_general_object(
    obj_conf, obj_conf_nm, multi=False, add_r_keys=[], add_p_keys=[]
):
    obj_required_keys = ["target"] + add_r_keys
    obj_possible_keys = obj_required_keys + ["params"] + add_p_keys
    if multi:
        for sub_obj_nm, sub_obj_conf in obj_conf.items():
            validate_keys(
                sub_obj_conf.keys(),
                obj_required_keys,
                obj_possible_keys,
                f"{obj_conf_nm}.{sub_obj_nm}",
            )
    else:
        validate_keys(
            obj_conf.keys(),
            obj_required_keys,
            obj_possible_keys,
            obj_conf_nm,
        )


def validate_conf(
    conf: OmegaConf,
    data_dir: str,
    devices: Sequence[int],
    logger: Logger = None,
    validate_datasets=True,
) -> None:
    do_train = "train" in conf.keys()
    do_val = "val" in conf.keys()

    # validate root configuration
    if True:
        root_required_keys = [
            "name",
            "datasets",
            "learner",
            "epochs",
        ]
        root_possible_keys = root_required_keys + [
            "augmentors",
            "losses",
            "optimizer",
            "lr_scheduler",
            "visualizers",
            "train",
            "val",
            "test",
            "checkpoints",
            "tollerence",
        ]
        validate_keys(conf.keys(), root_required_keys, root_possible_keys, "conf")
        if do_val and not do_train:
            raise AttributeError(
                "If 'val' configuration is defined, 'train' must also be defined"
            )
        if do_train and any([k not in conf.keys() for k in ["losses", "optimizer"]]):
            raise AttributeError(
                "'loss' and 'optimizer' must be defined if 'train' is defined"
            )

    # validate name
    assert type(conf.name) == str, "Configuration name must be a string"

    # validate datasets
    if True:
        if validate_datasets:
            datasets = load_datasets(conf)
            for k, ds in datasets.items():
                assert len(ds) > 0, f"Dataset '{k}' has length 0."

    # validate the learner configuration
    if True:
        learner_required_keys = ["target"]
        learner_possible_keys = learner_required_keys + [
            "params",
            "local_device_maps",
            "freeze",
        ]
        validate_keys(
            conf.learner.keys(),
            learner_required_keys,
            learner_possible_keys,
            "conf.learner",
        )
        if "local_device_maps" in conf.learner.keys():
            if len(conf.learner.local_device_maps) > len(devices):
                raise AttributeError(
                    f"The number of specified 'devices' ({len(devices)}) must be greater than or equal to the number of 'conf.learner.local_device_maps'({len(conf.learner.local_device_maps)})"
                )
        if conf.learner.target == "mt_pipe.src.util.learner_mux.LearnerMux":
            parallel_dataloader = True
            datapath_names = list(conf.learner.params.chldrn.keys())
        else:
            parallel_dataloader = False
            datapath_names = []

    # validate epochs
    assert (
        type(conf.epochs) == int and conf.epochs > 0
    ), "conf.epochs must be a positive integer"

    # validate augmentors
    if "augmentors" in conf.keys():
        validate_general_object(conf.augmentors, "conf.augmentors", True)
        # for aug_nm, aug_conf in conf.augmentors.items():
        #     aug_required_keys = ["target"]
        #     aug_possible_keys = aug_required_keys + ["params"]
        #     validate_keys(
        #         aug_conf.keys(),
        #         aug_required_keys,
        #         aug_possible_keys,
        #         f"conf.augmentors['{aug_nm}']",
        #     )

    # validate the losses configurations
    if "losses" in conf:
        validate_general_object(
            conf.losses, "conf.losses", True, add_p_keys=["local_device_maps"]
        )
        # loss_required_keys = ["target"]
        # loss_possible_keys = loss_required_keys + ["params", "local_device_maps"]

        # for loss_nm, loss_conf in conf.losses.items():
        #     validate_keys(
        #         loss_conf.keys(),
        #         loss_required_keys,
        #         loss_possible_keys,
        #         f"conf.losses.{loss_nm}",
        #     )

    # validate the optimizer configurations
    if "optimizer" in conf:
        validate_general_object(conf.optimizer, "conf.optimizer")
        # optimizer_required_keys = ["target"]
        # optimizer_possible_keys = optimizer_required_keys + ["params"]
        # validate_keys(
        #     conf.optimizer.keys(),
        #     optimizer_required_keys,
        #     optimizer_possible_keys,
        #     "conf.optimizer",
        # )

    # validate the lr_scheduler configurations
    if "lr_scheduler" in conf.keys():
        validate_general_object(conf.lr_scheduler, "conf.lr_scheduler")
        # lr_scheduler_required_keys = ["target"]
        # lr_scheduler_possible_keys = lr_scheduler_required_keys + ["params"]
        # validate_keys(
        #     conf.lr_scheduler.keys(),
        #     lr_scheduler_required_keys,
        #     lr_scheduler_possible_keys,
        #     "conf.lr_scheduler",
        # )

    if "visualizers" in conf.keys():
        validate_general_object(conf.visualizers, "conf.visualizers", True)

    def validate_loop(loop_conf, loop_conf_str):
        def validate_single_datapath(datapath_conf, datapath_conf_nm):
            datapath_required_keys = ["dataset", "loader_params"]
            datapath_possible_keys = datapath_required_keys + [
                "augmentor",
                "loss",
                "visualizer",
            ]
            validate_keys(
                datapath_conf.keys(),
                datapath_required_keys,
                datapath_possible_keys,
                datapath_conf_nm,
            )
            assert (
                "batch_size" in datapath_conf.loader_params
            ), f"'{datapath_conf_nm}.batch_size' is required"
            for k, v in zip(
                ["dataset", "augmentor", "loss", "visualizer"],
                ["datasets", "augmentors", "losses", "visualizers"],
            ):
                if k in datapath_conf:
                    assert (
                        datapath_conf[k] in getattr(conf, v).keys()
                    ), f"{datapath_conf_nm}.{k} must use an object defined under conf.{v}"

        if parallel_dataloader:
            assert all([k in datapath_names for k in loop_conf.keys()])
            for dp_nm, dp_conf in loop_conf.items():
                validate_single_datapath(dp_conf, f"{loop_conf_str}.{dp_nm}")
        else:
            validate_single_datapath(loop_conf, loop_conf_str)

    # validate train loop
    if "train" in conf:
        validate_loop(conf.train, "conf.train")

    # validate val loop
    if "val" in conf:
        validate_loop(conf.val, "conf.val")

    # validate test loop
    if "test" in conf:
        validate_loop(conf.test, "conf.test")

    # validate checkpoints
    if "checkpoints" in conf:
        assert type(conf.checkpoints) == ListConfig, "`conf.checkpoints` must be a list"
        for i, ckpt in enumerate(conf.checkpoints):
            assert set(ckpt.keys()) == {
                "name",
                "epoch",
            }, f"Error at conf.checkpoints[{i}]. A single checkpoint definition must and must only contain the keys ['name', 'epoch']"
            assert (
                type(ckpt.epoch) == int and ckpt.epoch > 0
            ), f"conf.checkpoints[{i}].epoch must be a positive integer"
            assert (
                type(ckpt.name) == str
            ), f"conf.checkpoints[{i}].name must be a string"

    # validate tollerence
    if "tollerence" in conf:
        assert (
            type(conf.tollerence) == int and conf.tollerence > 0
        ), "conf.tollerence must be a positive integer"

    if logger is not None:
        logger.info(f"Single task configuration for '{conf.name}' valid")
