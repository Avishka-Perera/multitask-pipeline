import torch
import os, shutil
from omegaconf import OmegaConf
from typing import Sequence, Dict, Literal
import yaml
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import pandas as pd
import omegaconf
from .logger import Logger
from .util import (
    yaml_loader,
    load_class,
    validate_keys,
    fix_list_len,
    load_config,
    load_states,
)
from .reporting import history_to_csv, history_to_img
from ..losses import ConcatLoss, BaseLoss
from ..evaluators import BaseEvaluator
from ..learners import BaseLearner
from ..datasets import BaseDataset
from ..constants import analysis_levels
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import gc
from functools import partial


def validate_conf(
    conf: omegaconf.OmegaConf,
    devices: Sequence[int],
    logger: Logger = None,
    validate_datasets=True,
) -> None:
    def validate_common():
        # validate root configuration
        root_required_keys = [
            "name",
            "data",
            "model",
            "loss",
            "optimizer",
            "train",
            "epochs",
        ]
        root_possible_keys = root_required_keys + [
            "tasks",
            "lr_scheduler",
            "val",
            "test",
            "checkpoints",
        ]
        validate_keys(conf.keys(), root_required_keys, root_possible_keys, "conf")

        # validate the model configuration
        model_required_keys = ["target"]
        model_possible_keys = model_required_keys + ["params", "local_device_maps"]
        validate_keys(
            conf.model.keys(),
            model_required_keys,
            model_possible_keys,
            "conf.model",
        )

        # validate the loss configurations start
        loss_required_keys = ["target"]
        loss_possible_keys = loss_required_keys + ["params", "local_device_maps"]

        def validate_loss_conf(
            loss_conf: omegaconf.dictconfig.DictConfig, loss_conf_name: str
        ) -> None:
            validate_keys(
                loss_conf.keys(),
                loss_required_keys,
                loss_possible_keys,
                loss_conf_name,
            )
            if "local_device_maps" in loss_conf.keys():
                if len(loss_conf.local_device_maps) > len(devices):
                    raise AttributeError(
                        f"The number of specified 'devices' ({len(devices)}) must be greater than or equal to the number of '{loss_conf_name}.local_device_maps'({len(loss_conf.local_device_maps)})"
                    )

        if "target" in (conf.loss):
            validate_loss_conf(conf.loss, "conf.loss")
        else:
            if type(conf.loss) == omegaconf.listconfig.ListConfig:
                availabe_losses = []
                for i, loss_conf in enumerate(conf.loss):
                    validate_loss_conf(loss_conf, f"conf.loss[{i}]")
                    if loss_conf.target in availabe_losses:
                        raise ValueError(
                            f"Same error cannot be repeated multiple times"
                        )
                    availabe_losses.append(loss_conf.target)
            else:
                loss_names = list(conf.loss.keys())
                if "loss" in conf.train:
                    assert all(
                        [nm in loss_names for nm in conf.train.loss]
                    ), "'conf.train.loss' must only specify losses defined at 'conf.loss'"
                if "val" in conf and "loss" in conf.val:
                    assert all(
                        [nm in loss_names for nm in conf.val.loss]
                    ), "'conf.val.loss' must only specify losses defined at 'conf.loss'"
                for nm, loss_conf in conf.loss.items():
                    validate_loss_conf(loss_conf, f"conf.loss.{nm}")
        # validate the loss configurations end

        # validate the optimizer configurations
        optimizer_required_keys = ["target"]
        optimizer_possible_keys = optimizer_required_keys + ["params"]
        validate_keys(
            conf.optimizer.keys(),
            optimizer_required_keys,
            optimizer_possible_keys,
            "conf.optimizer",
        )

        # validate the lr_scheduler configurations
        if "lr_scheduler" in conf.keys():
            lr_scheduler_required_keys = ["target"]
            lr_scheduler_possible_keys = lr_scheduler_required_keys + ["params"]
            validate_keys(
                conf.lr_scheduler.keys(),
                lr_scheduler_required_keys,
                lr_scheduler_possible_keys,
                "conf.lr_scheduler",
            )

        # validate the train configurations
        train_required_keys = ["loader_params"]
        train_possible_keys = train_required_keys + ["tollerance", "loss"]
        validate_keys(
            conf.train.keys(),
            train_required_keys,
            train_possible_keys,
            "conf.train",
        )

        # validate the val configurations
        if "val" in conf.keys():
            val_required_keys = ["loader_params"]
            val_possible_keys = val_required_keys + ["loss"]
            validate_keys(
                conf.val.keys(),
                val_required_keys,
                val_possible_keys,
                "conf.val",
            )

    def validate_single_task_conf() -> None:
        # validate the model configuration
        if "local_device_maps" in conf.model.keys():
            if len(conf.model.local_device_maps) > len(devices):
                raise AttributeError(
                    f"The number of specified 'devices' ({len(devices)}) must be greater than or equal to the number of 'conf.model.local_device_maps'({len(conf.model.local_device_maps)})"
                )

        # validate the data configurations
        data_required_keys = ["target", "params"]
        data_possible_keys = data_required_keys
        validate_keys(
            conf.data.keys(), data_required_keys, data_possible_keys, "conf.data"
        )
        data_param_required_keys = ["root"]
        data_param_possible_keys = data_param_required_keys + [
            "conf",
            "resize_wh",
            "dataset",
        ]
        validate_keys(
            conf.data.params.keys(),
            data_param_required_keys,
            data_param_possible_keys,
            "conf.data.params",
        )
        if "conf" in conf.data.params:
            # i.e., a ConcatSet
            data_param_required_keys = ["root"]
            data_param_possible_keys = data_param_required_keys + [
                "conf",
            ]
            validate_keys(
                conf.data.params.keys(),
                data_param_required_keys,
                data_param_possible_keys,
                "conf.data.params",
            )
            ds_conf_required_keys = ["target"]
            ds_conf_possible_keys = ds_conf_required_keys + [
                "split_mix",
                "params",
                "reps",
            ]
            for i, ds_conf in enumerate(conf.data.params.conf):
                validate_keys(
                    ds_conf.keys(),
                    ds_conf_required_keys,
                    ds_conf_possible_keys,
                    f"conf.data.params.conf[{i}]",
                )

        # start validate loss device_maps
        def validate_loss_devices(loss_conf, name):
            if "local_device_maps" in loss_conf.keys():
                if len(loss_conf.local_device_maps) > len(devices):
                    raise AttributeError(
                        f"The number of specified 'devices' ({len(devices)}) must be greater than or equal to the number of '{name}.local_device_maps'({len(loss_conf.local_device_maps)})"
                    )

        if type(conf.loss) == omegaconf.dictconfig.DictConfig:
            validate_loss_devices(conf.loss, "conf.loss")
        else:
            for i, loss_conf in enumerate(conf.loss):
                validate_loss_devices(loss_conf, f"conf.loss[{i}]")
        # end validate loss device_maps

        # validate datasets
        if validate_datasets:
            train_ds, val_ds, test_ds = load_datasets(
                conf.data, multi_task, do_val, do_test
            )
            assert len(train_ds) > 0, "The train_ds has length 0"
            assert not do_val or len(val_ds) > 0, "The val_ds has length 0"
            assert not do_test or len(test_ds) > 0, "The test_ds has length 0"

        if logger is not None:
            logger.info(f"Single task configuration for '{conf.name}' valid")

    def validate_multi_task_conf() -> None:
        tasks = conf.tasks

        # validate the model configuration
        if "local_device_maps" in conf.model.keys():
            for task in tasks:
                if task in conf.model.local_device_maps.keys() and (
                    len(conf.model.local_device_maps[task]) > len(devices)
                ):
                    raise AttributeError(
                        f"The number of specified 'devices' ({len(devices)}) must be greater than or equal to the number of 'conf.model.local_device_maps'({len(conf.model.local_device_maps[task])})"
                    )

        # validate the data configurations
        data_required_keys = tasks
        data_possible_keys = data_required_keys
        validate_keys(
            conf.data.keys(), data_required_keys, data_possible_keys, "conf.data"
        )
        for task in tasks:
            data_required_keys = ["target", "params"]
            data_possible_keys = data_required_keys
            validate_keys(
                conf.data[task].keys(),
                data_required_keys,
                data_possible_keys,
                f"conf.data.{task}",
            )
            data_param_required_keys = ["root"]
            data_param_possible_keys = data_param_required_keys + [
                "targets",
                "reps",
                "val_in_train",
            ]
            validate_keys(
                conf.data[task].params.keys(),
                data_param_required_keys,
                data_param_possible_keys,
                f"conf.data.{task}.params",
            )

        # validate loss device_maps
        if "local_device_maps" in conf.loss.keys():
            for task in tasks:
                if type(conf.loss) == omegaconf.dictconfig.DictConfig:
                    if task in conf.loss.local_device_maps and (
                        len(conf.loss.local_device_maps[task]) > len(devices)
                    ):
                        raise AttributeError(
                            f"The number of specified 'devices' ({len(devices)}) must be greater than or equal to the number of 'conf.loss.local_device_maps'({len(conf.loss.local_device_maps[task])})"
                        )
                else:
                    raise NotImplementedError()

        # validate datasets
        if validate_datasets:
            train_ds, val_ds, test_ds = load_datasets(
                conf.data, multi_task, do_val, do_test
            )
            for task in train_ds.keys():
                tds = train_ds[task]
                vds = val_ds[task]
                tsds = test_ds[task]
                assert len(tds) > 0, f"The train_ds of task '{task}' has length 0"
                assert (
                    not do_val or len(vds) > 0
                ), f"The val_ds of task '{task}' has length 0"
                assert (
                    not do_test or len(tsds) > 0
                ), f"The test_ds of task '{task}' has length 0"

        if logger is not None:
            logger.info(f"Multi task configuration for '{conf.name}' valid")

    multi_task = "tasks" in conf.keys()

    do_val = "val" in conf.keys()
    do_test = "test" in conf.keys()

    validate_common()
    if multi_task:
        validate_multi_task_conf()
    else:
        validate_single_task_conf()


def load_datasets(
    conf: omegaconf.OmegaConf, multi_task, do_val, do_test
) -> Dict[str, Dataset] | Dataset:
    if multi_task:
        tasks = list(conf.keys())
        for task in tasks:
            assert "target" in conf[task].keys()
        train_ds = {}
        val_ds = {}
        test_ds = {}
        for task in tasks:
            dataset_class = load_class(conf[task].target)
            train_ds[task] = dataset_class(**dict(conf[task].params), split="train")
            if do_val:
                val_ds[task] = dataset_class(**dict(conf[task].params), split="val")
            else:
                val_ds[task] = None
            if do_test:
                test_ds[task] = dataset_class(**dict(conf[task].params), split="test")
            else:
                test_ds[task] = None
    else:
        dataset_class = load_class(conf.target)
        train_ds = dataset_class(**dict(conf.params), split="train")
        if do_val:
            val_ds = dataset_class(**dict(conf.params), split="val")
        else:
            val_ds = None
        if do_test:
            test_ds = dataset_class(**dict(conf.params), split="test")
        else:
            test_ds = None

    return train_ds, val_ds, test_ds


class Trainer:
    already_trained_msg = "Training already done!"

    def _load_datasets(
        self,
    ) -> Sequence[BaseDataset] | BaseDataset:
        train_ds, val_ds, test_ds = load_datasets(
            self.conf.data, self.multi_task, self.do_val, self.do_test
        )

        return train_ds, val_ds, test_ds

    def _load_model(self):
        model_class = load_class(self.conf.model.target)

        # map the devices
        if "local_device_maps" in self.conf.model.keys():
            if self.multi_task:
                devices = {
                    task: (
                        [
                            self.devices[local_id]
                            for local_id in self.conf.model.local_device_maps[task]
                        ]
                        if task in self.conf.model.local_device_maps
                        else self.devices
                    )
                    for task in self.tasks
                }
            else:
                devices = [
                    self.devices[local_id]
                    for local_id in self.conf.model.local_device_maps
                ]
        else:
            if self.multi_task:
                devices = {task: self.devices for task in self.tasks}
            else:
                devices = self.devices

        # function to correct the devices
        def get_correct_device_lst(devices, device_cnt):
            if len(devices) < device_cnt:
                self.logger.warn(
                    f"Provided less number of devices ({len(devices)}) than expected ({device_cnt}). Fitting the model into a limited number of devices."
                )
                devices = fix_list_len(devices, device_cnt)
            elif len(devices) > device_cnt:
                self.logger.warn(
                    f"Too many devices specified ({len(devices)}) than expected ({device_cnt}). Trimming extra devices."
                )
                devices = fix_list_len(devices, device_cnt)
            return devices

        # correct the devices
        if self.multi_task:
            for task, task_devices in devices.items():
                devices[task] = get_correct_device_lst(
                    task_devices, model_class.device_count[task]
                )
        else:
            devices = get_correct_device_lst(devices, model_class.device_count)

        self.logger.info(f"Model: using devices: {devices}")
        if "params" in self.conf.model.keys():
            model: BaseLearner = model_class(
                **dict(self.conf.model.params), devices=devices, logger=self.logger
            )
        else:
            model: BaseLearner = model_class(devices=devices, logger=self.logger)

        if self.is_ddp:
            # self.model = DDP(model, find_unused_parameters=True)
            self.model = DDP(model)
        else:
            self.model = model

    def _load_model_weights(
        self, ckpt_path: str, ckpt_map_conf_path: str = None
    ) -> None:
        model = self.model.module if self.is_ddp else self.model
        sd = torch.load(ckpt_path)["model"]
        if ckpt_map_conf_path is None:
            model.load_state_dict(sd)
        else:
            with open(ckpt_map_conf_path) as handler:
                model_map_info = OmegaConf.create(yaml.load(handler, yaml.FullLoader))
            load_states(model, sd, model_map_info, self.logger)

    def _load_loss(self):
        def load_single_loss(loss_conf: omegaconf.dictconfig.DictConfig):
            loss_class = load_class(loss_conf.target)

            # map devices
            if "local_device_maps" in loss_conf.keys():
                if self.multi_task:
                    device = {
                        task: (
                            [
                                self.devices[local_id]
                                for local_id in loss_conf.local_device_maps[task]
                            ]
                            if task in loss_conf.local_device_maps
                            else self.devices
                        )[-1]
                        for task in self.tasks
                    }
                else:
                    device = [
                        self.devices[local_id]
                        for local_id in loss_conf.local_device_maps
                    ][-1]
            else:
                if self.multi_task:
                    device = {task: self.devices[-1] for task in self.tasks}
                else:
                    device = self.devices[-1]

            self.logger.info(f"Loss: using device: {device}")

            # create the loss object
            loss_params = dict(loss_conf.params) if "params" in loss_conf.keys() else {}
            loss = loss_class(
                **loss_params,
                device=device,
            )

            return loss

        if "target" in self.conf.loss:
            loss_fn: BaseLoss = load_single_loss(self.conf.loss)
            self.train_loss_fn = loss_fn
            self.val_loss_fn = loss_fn
        else:
            if type(self.conf.loss) == omegaconf.listconfig.ListConfig:
                loss_fns = {}
                for loss_conf in self.conf.loss:
                    loss_fns[loss_conf.target] = load_single_loss(loss_conf)
                loss_fn: BaseLoss = ConcatLoss(
                    device=-1, weight=1, logger=self.logger, conf=loss_fns
                )
                self.train_loss_fn = loss_fn
                self.val_loss_fn = loss_fn
            else:
                loss_names = list(self.conf.loss.keys())
                train_loss_fns = {}
                val_loss_fns = {}
                train_loss_names = (
                    self.conf.train.loss if "loss" in self.conf.train else loss_names
                )
                if self.do_val:
                    val_loss_names = (
                        self.conf.val.loss if "loss" in self.conf.val else loss_names
                    )
                else:
                    val_loss_names = {}
                for nm, loss_conf in self.conf.loss.items():
                    loss_fn = load_single_loss(loss_conf)
                    if nm in train_loss_names:
                        train_loss_fns[nm] = loss_fn
                    if nm in val_loss_names:
                        val_loss_fns[nm] = loss_fn
                self.train_loss_fn = ConcatLoss(
                    device=-1, weight=1, logger=self.logger, conf=train_loss_fns
                )
                self.val_loss_fn = ConcatLoss(
                    device=-1, weight=1, logger=self.logger, conf=val_loss_fns
                )

    def _load_training_objects(self):
        if "optimizer" in self.conf:
            optim_class = load_class(self.conf.optimizer.target)
            self.optimizer = optim_class(
                self.model.parameters(), **dict(self.conf.optimizer.params)
            )
        else:
            self.optimizer = Adam(self.model.parameters())
        self.logger.info(f"Using optimizer {self.optimizer.__class__.__name__}")

        if "lr_scheduler" in self.conf:
            scheduler_class = load_class(self.conf.lr_scheduler.target)
            self.lr_scheduler: LRScheduler = scheduler_class(
                self.optimizer, **dict(self.conf.lr_scheduler.params)
            )
            self.logger.info(f"Using scheduler {self.lr_scheduler.__class__.__name__}")
        else:
            self.lr_scheduler = None
            self.logger.info("Not using any scheduler")

    def _validate_conf(self):
        validate_conf(self.conf, self.devices, self.logger)

    # TODO: get rid of this
    def _fuse_multi_task(self):
        # Whenever flow and depth are found together, the new key will be depth_flow
        if "depth" in self.conf.tasks and "flow" in self.conf.tasks:
            self.logger.info("Found 'depth' and 'flow' tasks. Fusing into 'depth_flow'")

            # validate
            assert (
                self.conf.data.depth.params.root == self.conf.data.flow.params.root
            ), "Depth and Flow must have the same data source"
            assert (
                self.conf.data.depth.target == self.conf.data.flow.target
            ), "Depth and Flow must have the same dataset"
            assert (
                self.conf.train.loader_params.depth.batch_size
                == self.conf.train.loader_params.flow.batch_size
            ), "Batch sizes of flow and depth must be equal"
            assert (
                self.conf.val.loader_params.depth.batch_size
                == self.conf.val.loader_params.flow.batch_size
            ), "Batch sizes of flow and depth must be equal"
            assert (
                "local_device_maps" not in self.conf.model
                or "flow" not in self.conf.model.local_device_maps
                or (
                    self.conf.model.local_device_maps.flow
                    == self.conf.model.local_device_maps.depth
                )
            ), "local_device_maps (model) of flow and depth must be equal"
            assert (
                "local_device_maps" not in self.conf.loss
                or "flow" not in self.conf.loss.local_device_maps
                or self.conf.loss.local_device_maps.flow
                == self.conf.loss.local_device_maps.depth
            ), "local_device_maps (loss) of flow and depth must be equal"

            # update the tasks
            self.conf.tasks.append("depth_flow")
            self.conf.tasks.remove("depth"), self.conf.tasks.remove("flow")

            # update the data.<task>
            self.conf.data["depth_flow"] = self.conf.data["flow"]
            self.conf.data.pop("depth"), self.conf.data.pop("flow")

            # update the batch_sizes
            self.conf.train.loader_params[
                "depth_flow"
            ] = self.conf.train.loader_params.depth
            self.conf.train.loader_params.pop(
                "depth"
            ), self.conf.train.loader_params.pop("flow")
            self.conf.val.loader_params[
                "depth_flow"
            ] = self.conf.val.loader_params.depth
            self.conf.val.loader_params.pop("depth"), self.conf.val.loader_params.pop(
                "flow"
            )

            # update the loss
            self.conf.loss.params.tasks.remove("depth")
            self.conf.loss.params.tasks.remove("flow")
            self.conf.loss.params.tasks.append("depth_flow")

            # update local_device_maps (loss)
            if ("local_device_maps" in self.conf.loss) and (
                "flow" in self.conf.loss.local_device_maps.keys()
            ):
                self.conf.loss.local_device_maps[
                    "depth_flow"
                ] = self.conf.loss.local_device_maps.flow
                new_device_map = omegaconf.OmegaConf.to_container(
                    self.conf.loss.local_device_maps
                )
                new_device_map.pop("flow")
                new_device_map.pop("depth")
                new_device_map = omegaconf.OmegaConf.create(new_device_map)
                self.conf.loss.local_device_maps = new_device_map

            # update local_device_maps (model)
            if ("local_device_maps" in self.conf.model) and (
                "flow" in self.conf.model.local_device_maps
            ):
                self.conf.model.local_device_maps[
                    "depth_flow"
                ] = self.conf.model.local_device_maps.flow
                new_device_map = omegaconf.OmegaConf.to_container(
                    self.conf.model.local_device_maps
                )
                new_device_map.pop("flow")
                new_device_map.pop("depth")
                new_device_map = omegaconf.OmegaConf.create(new_device_map)
                self.conf.model.local_device_maps = new_device_map

            # update the learner
            self.conf.model.params.tasks.remove("depth")
            self.conf.model.params.tasks.remove("flow")
            self.conf.model.params.tasks.append("depth_flow")

    def _load_evaluator(self) -> None:
        self.evaluators: Dict[str, BaseEvaluator] = {}
        if self.do_test:
            for eval_nm, eval_conf in self.conf.test.evaluators.items():
                eval_class = load_class(eval_conf.target)
                params = dict(eval_conf.params) if "params" in eval_conf else {}
                self.evaluators[eval_nm] = eval_class(
                    rank=self.rank,
                    world_size=self.world_size,
                    logger=self.logger,
                    **params,
                )

    def __init__(
        self,
        conf: str | Dict | omegaconf.dictconfig.DictConfig,
        weights_conf: Dict[str, str],
        devices: Sequence[str | int],
        rank: int = None,
        world_size: int = 1,
        logger: Logger = None,
        analysis_level: Literal[*analysis_levels] = 0,
    ) -> None:
        if type(conf) == str:
            with open(conf) as handler:
                dict_conf = yaml.load(handler, yaml_loader)
        elif type(conf) == dict:
            dict_conf == conf
        elif type(conf) == omegaconf.dictconfig.DictConfig:
            dict_conf = dict(conf)
        else:
            raise ValueError("Unexpected type provided for conf")
        self.conf = omegaconf.OmegaConf.create(dict_conf)
        self.devices = devices
        self.is_ddp = rank is not None
        self.rank = rank
        self.world_size = world_size
        self.do_out = self.rank == 0 or self.rank is None
        self.multi_task = "tasks" in self.conf.keys()
        self.logger = Logger(0, rank) if logger is None else logger
        self._validate_conf()
        if self.multi_task:
            self._fuse_multi_task()
            self.tasks = self.conf.tasks
        self.do_val = "val" in self.conf.keys()
        self.do_test = "test" in self.conf.keys()
        if not self.do_val:
            self.logger.warn(
                "No validation configuration detected. Validation loops will be skipped"
            )
        self.train_ds, self.val_ds, self.test_ds = self._load_datasets()
        self._load_model()
        if weights_conf["ckpt_path"] is not None:
            self._load_model_weights(
                weights_conf["ckpt_path"], weights_conf["ckpt_map_conf_path"]
            )
        self._load_loss()
        self._load_evaluator()
        self._load_training_objects()
        self.ckpts = (
            {ckpt.epoch: ckpt.name for ckpt in self.conf.checkpoints}
            if "checkpoints" in self.conf.keys()
            else {}
        )
        self.analysis_level = analysis_level

    def _unpack_losspack_recursive(self, loss_pack, lead=None):
        nm_loss_dict = {}
        for nm, val in loss_pack.items():
            hier_nm = nm if lead is None else f"{lead}:{nm}"
            if type(val) != dict:
                if nm != "tot":
                    nm_loss_dict[hier_nm] = val
            else:
                nested_dict = self._unpack_losspack_recursive(val, hier_nm)
                nm_loss_dict = {**nm_loss_dict, **nested_dict}
        return nm_loss_dict

    def _plot_loss_bacth(
        self,
        loss_pack: Dict[str, torch.Tensor],
        stage: Literal["Train", "Val"],
        batch_id,
        epoch,
    ) -> None:
        card_nm_plt = f"EPOCH: {epoch}"
        loss_pack = self._unpack_losspack_recursive(loss_pack)
        for nm, loss in loss_pack.items():
            loss = loss.cpu().item()
            cat_plt = f"Loss (per epoch):{stage}:{nm}"
            cat_acc = f"Loss Analysis (job wide):{stage}"
            card_nm_acc = nm
            self.logger.plot(cat_plt, card_nm_plt, loss, batch_id)
            self.logger.accumulate(cat_acc, card_nm_acc, loss)

    def val_step(
        self,
        batch: Sequence[torch.Tensor] | Dict[str, Sequence[torch.Tensor]],
        epoch: int,
        batch_id: int,
    ) -> torch.Tensor:
        out = self.model(batch)
        loss_pack = self.val_loss_fn(out, batch)
        tot_loss = loss_pack["tot"]

        if self.analysis_level > 0 and self.do_out:
            self._plot_loss_bacth(loss_pack, "Val", batch_id, epoch)

        return tot_loss

    def train_step(
        self,
        batch: Sequence[torch.Tensor] | Dict[str, Sequence[torch.Tensor]],
        epoch: int,
        batch_id: int,
    ) -> float:
        self.optimizer.zero_grad()
        out = self.model(batch)
        loss_pack = self.train_loss_fn(out, batch)
        tot_loss = loss_pack["tot"]
        tot_loss.backward()
        self.logger.batch_step(analyze_grads=self.analysis_level > 1)
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

        if self.analysis_level > 0 and self.do_out:
            self._plot_loss_bacth(loss_pack, "Train", batch_id, epoch)

        tot_loss = tot_loss.detach().cpu().item()
        return tot_loss

    def _save_ckpt(self, epoch, output_path, status, history=None):
        # checkpoints
        ckpt_path = os.path.join(output_path, "ckpts", status + ".ckpt")
        model_state = (
            self.model.module.state_dict()
            if type(self.model) == DDP
            else self.model.state_dict()
        )
        optim_state = self.optimizer.state_dict()
        lr_scheduler_state = (
            None if self.lr_scheduler is None else self.lr_scheduler.state_dict()
        )
        ckpts = {
            "model": model_state,
            "optimizer": optim_state,
            "lr_scheduler": lr_scheduler_state,
            "epoch": epoch,
        }
        torch.save(ckpts, ckpt_path)

        # history
        if history is not None:
            history_to_img(history, os.path.join(output_path, "history.jpg"))
            history_to_csv(history, os.path.join(output_path, "history.csv"))

    def _load_ckpt(self, path):
        self.logger.info(f"Loading checkpoints from {path}")
        ckpts = torch.load(path)
        model_state = ckpts["model"]
        optim_state = ckpts["optimizer"]
        lr_scheduler_state = ckpts["lr_scheduler"]
        epoch = ckpts["epoch"]
        if type(self.model) == DDP:
            self.model.module.load_state_dict(model_state)
        else:
            self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optim_state)
        if self.lr_scheduler is not None and lr_scheduler_state is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_state)
        return epoch

    def _init_output(self, output_path, run_name):
        # make output directories
        if run_name is None:
            output_path = os.path.abspath(
                os.path.join(output_path, str(self.conf.name))
            )
            if os.path.exists(output_path):
                id = 0
                available_runs = os.listdir(output_path)
                while f"run{id}" in available_runs:
                    id += 1
                output_path = os.path.join(output_path, f"run{id}")
            else:
                output_path = os.path.join(output_path, "run0")
        else:
            output_path = os.path.abspath(os.path.join(output_path, run_name))
            if os.path.exists(output_path) and self.do_out:
                self.logger.warn(f"Removing existing files at '{output_path}'")
                shutil.rmtree(output_path)

        if self.is_ddp:
            dist.barrier()

        if self.do_out:
            os.makedirs(os.path.join(output_path, "ckpts"))
            # dump configs
            with open(os.path.join(output_path, "configuration.yaml"), "w") as file:
                yaml.dump(OmegaConf.to_container(self.conf), file)

        return output_path

    def _init_fit(
        self,
        output_path,
        run_name: str,
        resume_dir: str = None,
        force_resume: bool = False,
    ):
        # resume if ckpts are available
        if resume_dir is not None:
            assert (
                run_name is None
            ), "'run_name' should not be specified with 'resume_dir'"
            if not os.path.exists(resume_dir):
                raise FileNotFoundError(
                    f"Provided resume directory ({resume_dir}) dose not exist"
                )

            old_config = load_config(os.path.join(resume_dir, "configuration.yaml"))
            if old_config != self.conf:
                if not force_resume:
                    raise AssertionError(
                        f"The existing config file dose not match the provided configuration"
                    )

            output_path = resume_dir

            # load and init trackers
            exp_ckpt_path = os.path.join(output_path, "ckpts", "final.ckpt")
            if os.path.exists(exp_ckpt_path):
                start_epoch = self._load_ckpt(exp_ckpt_path) + 1
                history = pd.read_csv(os.path.join(output_path, "history.csv"))
                min_loss = history["val_loss"].min()
                best_epoch = history["val_loss"].argmin()
                history = history.to_dict()
                del history["epoch"]
                history["train_loss"] = list(history["train_loss"].values())
                history["val_loss"] = list(history["val_loss"].values())
            else:
                # init trackers
                start_epoch = 0
                history = (
                    {"train_loss": [], "val_loss": []}
                    if self.do_val
                    else {"train_loss": []}
                )
                min_loss = float("inf")
                best_epoch = -1
        else:
            output_path = self._init_output(output_path, run_name)

            # init trackers
            start_epoch = 0
            history = (
                {"train_loss": [], "val_loss": []}
                if self.do_val
                else {"train_loss": []}
            )
            min_loss = float("inf")
            best_epoch = -1

        if self.do_out:
            model = self.model.module if self.is_ddp else self.model
            self.logger.init_plotter(output_path, model)

            if self.do_test:
                for eval in self.evaluators.values():
                    eval.set_out_path(os.path.join(output_path, "results"))

        return output_path, min_loss, start_epoch, best_epoch, history

    def _get_adjusted_datasets(self, mock_batch_count: Sequence[int]):
        assert len(mock_batch_count) in [1, 3]
        train_mock_batch_count = mock_batch_count[0]
        if len(mock_batch_count) == 3:
            val_mock_batch_count = mock_batch_count[1]
            test_mock_batch_count = mock_batch_count[2]
        else:
            val_mock_batch_count = train_mock_batch_count
            test_mock_batch_count = train_mock_batch_count

        if self.multi_task:
            assert (not self.do_val) or isinstance(
                self.val_ds, Dict
            ), "val_ds should also be a Dict"
            assert all(
                [
                    "batch_size" in self.conf.train.loader_params[task].keys()
                    for task in self.conf.tasks
                ]
            ), "conf.train.loader_params.<task> must contain the 'batch_size'"
            assert all(
                [
                    "batch_size" in self.conf.val.loader_params[task].keys()
                    for task in self.conf.tasks
                ]
            ), "conf.val.loader_params.<task> must contain the 'batch_size'"
            assert (not self.do_val) or (
                len(self.train_ds) == len(self.val_ds)
            ), "Number of validation datasets should be equal to the train datasets"

        def get_mock_ds(split_conf, self_ds, mock_batch_count):
            return (
                {
                    k: Subset(
                        ds,
                        range(
                            min(
                                split_conf.loader_params[k].batch_size
                                * mock_batch_count
                                * self.world_size,
                                len(ds),
                            )
                        ),
                    )
                    for (k, ds) in self_ds.items()
                }
                if self.multi_task
                else Subset(
                    self_ds,
                    range(
                        min(
                            split_conf.loader_params.batch_size
                            * mock_batch_count
                            * self.world_size,
                            len(self_ds),
                        )
                    ),
                )
            )

        if train_mock_batch_count == -1:
            train_ds = self.train_ds
        else:
            train_ds = get_mock_ds(
                self.conf.train, self.train_ds, train_mock_batch_count
            )
        val_ds = self.val_ds
        if self.do_val:
            if val_mock_batch_count != -1:
                val_ds = get_mock_ds(self.conf.val, self.val_ds, val_mock_batch_count)
        test_ds = self.test_ds
        if self.do_test:
            if test_mock_batch_count != -1:
                test_ds = get_mock_ds(
                    self.conf.test, self.test_ds, test_mock_batch_count
                )

        return train_ds, val_ds, test_ds

    def _get_dataloaders(self, train_ds, val_ds, test_ds):
        if self.multi_task:
            # TODO: loader params for multi task
            def get_loader(dss, split_conf):
                samplers = (
                    {
                        k: DistributedSampler(ds, self.world_size, self.rank)
                        for k, ds in dss.items()
                    }
                    if self.is_ddp
                    else {k: None for k, ds in dss.items()}
                )
                loaders = {
                    k: (
                        DataLoader(
                            ds,
                            **dict(split_conf.loader_params[k]),
                            sampler=samplers[k],
                        )
                        if self.is_ddp
                        else DataLoader(ds, **dict(split_conf.loader_params[k]))
                    )
                    for k, ds in dss.items()
                }
                return loaders, samplers

            train_dl, train_sampler = get_loader(train_ds, self.conf.train)
            if self.do_val:
                val_dl, val_sampler = get_loader(val_ds, self.conf.val)
            else:
                val_dl = val_sampler = val_ds  # {task1: None, task2: None, ...}
            if self.do_test:
                test_dl, test_sampler = get_loader(test_ds, self.conf.test)
            else:
                test_dl = test_sampler = test_ds  # {task1: None, task2: None, ...}
        else:

            def process_loader_params_single_task(loader_params: OmegaConf, ds) -> Dict:
                new_loader_params = OmegaConf.to_container(loader_params)
                if "collate_fn" in loader_params:
                    collate_fn_conf = loader_params.pop("collate_fn")
                    func = load_class(collate_fn_conf.target)
                    if "params" in collate_fn_conf:
                        new_func_params = {}
                        for k, v in collate_fn_conf.params.items():
                            if hasattr(v, "__iter__") and "target" in v:
                                # i.e., its a class object
                                cls = load_class(v.target)
                                params = dict(v.params) if "params" in v else {}
                                new_func_params[k] = cls(**params)
                            else:
                                new_func_params[k] = v
                        new_loader_params["collate_fn"] = partial(
                            func, **new_func_params
                        )
                    else:
                        new_loader_params["collate_fn"] = func

                if self.is_ddp:
                    if "sampler" in loader_params:
                        sampler_class = load_class(loader_params.sampler.target)
                    else:
                        sampler_class = DistributedSampler
                    new_loader_params["sampler"] = sampler_class(
                        ds, self.world_size, self.rank
                    )
                else:
                    if "sampler" in loader_params:
                        raise NotImplementedError(
                            "Custom sampler usage is not implemented for non-DDP setups"
                        )

                return new_loader_params

            def get_loader(ds, split_conf):
                params = process_loader_params_single_task(split_conf.loader_params, ds)
                sampler = params["sampler"] if "sampler" in params else None
                loader = DataLoader(ds, **dict(params))
                return loader, sampler

            train_dl, train_sampler = get_loader(train_ds, self.conf.train)
            if self.do_val:
                val_dl, val_sampler = get_loader(val_ds, self.conf.val)
            else:
                val_dl = val_sampler = val_ds  # None
            if self.do_test:
                test_dl, test_sampler = get_loader(test_ds, self.conf.test)
            else:
                test_dl = test_sampler = test_ds  # None

        # validate the dataloaders
        if self.multi_task:
            for k, dl in train_dl.items():
                if len(dl) == 0:
                    raise AttributeError(f"'{k}', 'train' dataloader has length 0")
            if self.do_val:
                for k, dl in val_dl.items():
                    if len(dl) == 0:
                        raise AttributeError(f"'{k}', 'val' dataloader has length 0")
            if self.do_test:
                for k, dl in test_dl.items():
                    if len(dl) == 0:
                        raise AttributeError(f"'{k}', 'test' dataloader has length 0")
        else:
            if len(train_dl) == 0:
                raise AttributeError("'train' dataloader has length 0")
            if val_dl is not None and (len(val_dl) == 0):
                raise AttributeError("'val' dataloader has length 0")
            if test_dl is not None and (len(test_dl) == 0):
                raise AttributeError("'test' dataloader has length 0")

        return ((train_dl, val_dl, test_dl), (train_sampler, val_sampler, test_sampler))

    def _train_loop(self, train_batch_count, show_pbar, epoch, train_dl, dl_keys):
        # train loop
        self.model.train()
        train_losses = []
        with tqdm(
            total=train_batch_count,
            disable=not (self.logger.display_info and show_pbar and self.do_out),
            desc="Training",
        ) as pbar:
            if not show_pbar:
                self.logger.info("Training")

            def process_batch(batch, batch_id):
                self.optimizer.zero_grad()
                train_loss = self.train_step(batch, epoch, batch_id)
                if self.do_out:
                    pbar.set_postfix(loss=train_loss)
                    pbar.update(1)
                    self.logger.plot(
                        "Loss (per epoch): Train",
                        f"EPOCH: {epoch}",
                        train_loss,
                        batch_id,
                    )
                train_losses.append(train_loss)

            if self.multi_task:
                for batch_id, batch in enumerate(zip(*train_dl)):
                    batch = {k: v for (k, v) in zip(dl_keys, batch)}
                    process_batch(batch, batch_id)
            else:
                for batch_id, batch in enumerate(train_dl):
                    process_batch(batch, batch_id)

        train_loss = sum(train_losses) / len(train_losses)

        return train_loss

    def _val_loop(self, val_batch_count, show_pbar, epoch, val_dl, dl_keys):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            with tqdm(
                total=val_batch_count,
                disable=not (self.logger.display_info and show_pbar and self.do_out),
                desc="Validating",
            ) as pbar:
                if not show_pbar:
                    self.logger.info("Validating")

                def process_batch(batch, batch_id, val_loss):
                    val_loss += self.val_step(batch, epoch, batch_id)
                    cur_val_loss = val_loss / (batch_id + 1)
                    if self.do_out:
                        pbar.set_postfix(loss=cur_val_loss.detach().cpu().item())
                        pbar.update(1)
                        self.logger.plot(
                            "Loss (per epoch): Val",
                            f"EPOCH: {epoch}",
                            val_loss,
                            batch_id,
                        )
                    return val_loss

                if self.multi_task:
                    for batch_id, batch in enumerate(zip(*val_dl)):
                        batch = {k: v for (k, v) in zip(dl_keys, batch)}
                        val_loss = process_batch(batch, batch_id, val_loss)
                else:
                    for batch_id, batch in enumerate(val_dl):
                        val_loss = process_batch(batch, batch_id, val_loss)
                val_loss /= val_batch_count

        if self.is_ddp:
            dist.all_reduce(val_loss, dist.ReduceOp.SUM)
            val_loss = (
                val_loss.detach().cpu().item() / self.world_size
            )  # since dist.ReduceOp.AVG is not available with the default backend (Gloo)
        else:
            val_loss = val_loss.detach().cpu().item()

        return val_loss

    def _test_loop(self, show_pbar: bool, test_dl: DataLoader) -> None:
        self.model.eval()
        with torch.no_grad():
            with tqdm(
                total=len(test_dl),
                disable=not (self.logger.display_info and show_pbar and self.do_out),
                desc="Testing",
            ) as pbar:
                if not show_pbar:
                    self.logger.info("Testing")
                for batch in test_dl:
                    out = self.model(batch)
                    for eval in self.evaluators.values():
                        eval.register(batch=batch, out=out)
                    pbar.update()

                for eval in self.evaluators.values():
                    eval.output()

    def _get_fit_info(self, mock_epoch_count, train_dl, val_dl):
        tollerance = self.conf.train.tollerance
        train_batch_count = (
            min([len(dl) for dl in train_dl.values()])
            if self.multi_task
            else len(train_dl)
        )
        if self.do_val:
            val_batch_count = (
                min([len(dl) for dl in val_dl.values()])
                if self.multi_task
                else len(val_dl)
            )
        else:
            val_batch_count = None

        if self.multi_task:
            dl_keys = train_dl.keys()
            train_dl = train_dl.values()
            val_dl = val_dl.values()
        else:
            dl_keys = None
        epochs = mock_epoch_count if mock_epoch_count > 0 else self.conf.epochs

        return (
            tollerance,
            train_batch_count,
            val_batch_count,
            dl_keys,
            train_dl,
            val_dl,
            epochs,
        )

    def fit(
        self,
        output_path: str = "out",
        run_name: str = None,
        mock_batch_count: Sequence[int] = [-1],
        mock_epoch_count: int = -1,
        resume_dir: str = None,
        force_resume: bool = False,
        show_pbar=True,
    ) -> None:
        output_path, min_loss, start_epoch, best_epoch, history = self._init_fit(
            output_path, run_name, resume_dir, force_resume
        )

        if start_epoch >= self.conf.epochs - 1:
            self.logger.info(self.already_trained_msg)
            return

        train_ds, val_ds, test_ds = self._get_adjusted_datasets(mock_batch_count)
        loaders, samplers = self._get_dataloaders(train_ds, val_ds, test_ds)
        train_dl, val_dl, test_dl = loaders
        train_sampler, val_sampler, test_sampler = samplers

        best_ckpt_path = os.path.join(output_path, "ckpts", "best.ckpt")
        final_ckpt_path = os.path.join(output_path, "ckpts", "final.ckpt")
        self.logger.info(f"Saving outputs to {output_path}")

        (
            tollerance,
            train_batch_count,
            val_batch_count,
            dl_keys,
            train_dl,
            val_dl,
            epochs,
        ) = self._get_fit_info(mock_epoch_count, train_dl, val_dl)

        def set_samplers_epoch(epoch):
            for smpl_l1 in (train_sampler, val_sampler, test_sampler):
                if type(smpl_l1) == dict:
                    for smpl_l2 in smpl_l1.values():
                        if smpl_l2 is not None:
                            smpl_l2.set_epoch(epoch)
                else:
                    if smpl_l1 is not None:
                        smpl_l1.set_epoch(epoch)

        for epoch in range(start_epoch, epochs):
            self.logger.info(
                f"---- EPOCH {str(epoch+1).rjust(len(str(epochs)), '0')}/{epochs} ----"
            )
            set_samplers_epoch(epoch)

            train_loss = self._train_loop(
                train_batch_count, show_pbar, epoch, train_dl, dl_keys
            )
            torch.cuda.empty_cache()

            if self.do_val:
                val_loss = self._val_loop(
                    val_batch_count, show_pbar, epoch, val_dl, dl_keys
                )
                torch.cuda.empty_cache()
            else:
                val_loss = None

            if self.is_ddp:
                dist.barrier()

            # save extra checkpoints if specified
            if epoch in self.ckpts.keys():
                name = self.ckpts[epoch].rstrip(".ckpt")
                self.logger.info(
                    f"Saving additional checkpoint '{name}' at {output_path}"
                )
                self._save_ckpt(epoch, output_path, name)

            # logging
            history["train_loss"].append(train_loss)
            if self.do_val:
                history["val_loss"].append(val_loss)
            if self.do_out:
                self.logger.plot("Loss (job wide)", "Train", train_loss, epoch)
                if self.do_val:
                    self.logger.plot("Loss (job wide)", "Val", val_loss, epoch)
                self.logger.step(epoch, analyze_grad=self.analysis_level > 1)

            # evaluation of the epoch performance
            if self.do_val and (val_loss < min_loss):
                min_loss = val_loss
                best_epoch = epoch
                if self.do_out:
                    self._save_ckpt(epoch, output_path, "best", history)
                    shutil.copy(best_ckpt_path, final_ckpt_path)
            else:
                if self.do_out:
                    self._save_ckpt(epoch, output_path, "final", history)
                if self.do_val:
                    if epoch - best_epoch > tollerance and tollerance >= 0:
                        self.logger.info(
                            f"Stopping early at epoch {epoch}. Best epoch found at {best_epoch} with {tollerance} tollerance"
                        )
                        break

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            gc.collect(0)
            gc.collect(1)
            gc.collect(2)
            torch.cuda.empty_cache()

        dist.barrier()

        if self.do_test:
            if os.path.exists(best_ckpt_path):
                sd = torch.load(best_ckpt_path)["model"]
            else:
                sd = torch.load(final_ckpt_path)["model"]
            if self.is_ddp:
                self.model.module.load_state_dict(sd)
            else:
                self.model.load_state_dict(sd)
            self._test_loop(show_pbar=show_pbar, test_dl=test_dl)

        self.logger.info(f"Single-staged training successful!\n")
