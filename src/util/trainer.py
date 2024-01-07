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
from ..util.data import ParallelDataLoader
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
        def validate_single_datapath(data_conf, nm):
            data_required_keys = ["target", "params"]
            data_possible_keys = data_required_keys
            validate_keys(data_conf.keys(), data_required_keys, data_possible_keys, nm)
            data_param_required_keys = ["root"]
            data_param_possible_keys = data_param_required_keys + [
                "conf",
                "resize_wh",  # TODO: move this to the conf or params itself
                "dataset",
            ]
            validate_keys(
                data_conf.params.keys(),
                data_param_required_keys,
                data_param_possible_keys,
                f"{nm}.params",
            )
            if "conf" in data_conf.params:
                # i.e., a ConcatSet
                data_param_required_keys = ["root"]
                data_param_possible_keys = data_param_required_keys + [
                    "conf",
                ]
                validate_keys(
                    data_conf.params.keys(),
                    data_param_required_keys,
                    data_param_possible_keys,
                    f"{nm}.params",
                )
                ds_conf_required_keys = ["target"]
                ds_conf_possible_keys = ds_conf_required_keys + [
                    "split_mix",
                    "params",
                    "reps",
                ]
                for i, ds_conf in enumerate(data_conf.params.conf):
                    validate_keys(
                        ds_conf.keys(),
                        ds_conf_required_keys,
                        ds_conf_possible_keys,
                        f"{nm}.params.conf[{i}]",
                    )

        if "target" in conf.data:
            validate_single_datapath(conf.data, "conf.data")
        else:
            # i.e., multi datapaths
            for nm, sub_conf in conf.data.items():
                validate_single_datapath(sub_conf, f"conf.data.{nm}")

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
            train_ds, val_ds, test_ds = load_datasets(conf.data, do_val, do_test)
            assert len(train_ds) > 0, "The train_ds has length 0"
            assert not do_val or len(val_ds) > 0, "The val_ds has length 0"
            assert not do_test or len(test_ds) > 0, "The test_ds has length 0"

        if logger is not None:
            logger.info(f"Single task configuration for '{conf.name}' valid")

    do_val = "val" in conf.keys()
    do_test = "test" in conf.keys()

    # TODO: bring the content of these functions out
    validate_common()
    validate_single_task_conf()


def load_datasets(
    conf: omegaconf.OmegaConf, do_val, do_test
) -> Sequence[Dict[str, Dataset]] | Sequence[Dataset]:
    def load_single_dataset(conf):
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

    if "target" in conf:
        return load_single_dataset(conf)
    else:
        train_dss, val_dss, test_dss = {}, {}, {}
        for nm, sub_conf in conf.items():
            train_ds, val_ds, test_ds = load_single_dataset(sub_conf)
            train_dss[nm] = train_ds
            val_dss[nm] = val_ds
            test_dss[nm] = test_ds

        return train_dss, val_dss, test_dss


class Trainer:
    already_trained_msg = "Training already done!"

    def _load_datasets(
        self,
    ) -> Sequence[BaseDataset] | BaseDataset:
        train_ds, val_ds, test_ds = load_datasets(
            self.conf.data, self.do_val, self.do_test
        )

        return train_ds, val_ds, test_ds

    def _load_model(self):
        model_class = load_class(self.conf.model.target)

        # map the devices
        if "local_device_maps" in self.conf.model.keys():
            devices = [
                self.devices[local_id] for local_id in self.conf.model.local_device_maps
            ]
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
        devices = get_correct_device_lst(devices, model_class.device_count)

        self.logger.info(f"Model: using devices: {devices}")
        if "params" in self.conf.model.keys():
            model: BaseLearner = model_class(
                **dict(self.conf.model.params), devices=devices
            )
        else:
            model: BaseLearner = model_class(devices=devices)

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
                device = [
                    self.devices[local_id] for local_id in loss_conf.local_device_maps
                ][-1]
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

        loss_fn: BaseLoss = load_single_loss(self.conf.loss)
        self.train_loss_fn = loss_fn
        self.val_loss_fn = loss_fn

        # TODO: resolve this. Automatic concatloss or defined concat loss?
        # if "target" in self.conf.loss:
        #     loss_fn: BaseLoss = load_single_loss(self.conf.loss)
        #     self.train_loss_fn = loss_fn
        #     self.val_loss_fn = loss_fn
        # else:
        #     if type(self.conf.loss) == omegaconf.listconfig.ListConfig:
        #         loss_fns = {}
        #         for loss_conf in self.conf.loss:
        #             loss_fns[loss_conf.target] = load_single_loss(loss_conf)
        #         loss_fn: BaseLoss = ConcatLoss(
        #             device=-1, weight=1, logger=self.logger, conf=loss_fns
        #         )
        #         self.train_loss_fn = loss_fn
        #         self.val_loss_fn = loss_fn
        #     else:
        #         loss_names = list(self.conf.loss.keys())
        #         train_loss_fns = {}
        #         val_loss_fns = {}
        #         train_loss_names = (
        #             self.conf.train.loss if "loss" in self.conf.train else loss_names
        #         )
        #         if self.do_val:
        #             val_loss_names = (
        #                 self.conf.val.loss if "loss" in self.conf.val else loss_names
        #             )
        #         else:
        #             val_loss_names = {}
        #         for nm, loss_conf in self.conf.loss.items():
        #             loss_fn = load_single_loss(loss_conf)
        #             if nm in train_loss_names:
        #                 train_loss_fns[nm] = loss_fn
        #             if nm in val_loss_names:
        #                 val_loss_fns[nm] = loss_fn
        #         self.train_loss_fn = ConcatLoss(
        #             device=-1, weight=1, logger=self.logger, conf=train_loss_fns
        #         )
        #         self.val_loss_fn = ConcatLoss(
        #             device=-1, weight=1, logger=self.logger, conf=val_loss_fns
        #         )

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
        self.logger = Logger(0, rank) if logger is None else logger
        self._validate_conf()
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
            if loss is not None:
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

        def get_mock_ds(split_conf, self_ds, mock_batch_count):
            def get(ds, batch_size):
                return Subset(
                    ds,
                    range(
                        min(
                            batch_size * mock_batch_count * self.world_size,
                            len(ds),
                        )
                    ),
                )

            if type(self_ds) == dict:
                return {
                    nm: get(ds, split_conf.loader_params[nm].batch_size)
                    for nm, ds in self_ds.items()
                }
            else:
                return get(self_ds, split_conf.loader_params.batch_size)

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
        def process_loader_params(loader_params: OmegaConf, ds) -> Dict:
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
                    new_loader_params["collate_fn"] = partial(func, **new_func_params)
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

        def get_loader(dss, split_conf):
            def get_single_loader(ds, loader_params):
                params = process_loader_params(loader_params, ds)
                sampler = params["sampler"] if "sampler" in params else None
                loader = DataLoader(ds, **dict(params))
                return loader, sampler

            if type(dss) == dict:
                samplers, dls = [], []
                for nm, ds in dss.items():
                    loader, sampler = get_single_loader(
                        ds, split_conf.loader_params[nm]
                    )
                    dls.append(loader)
                    samplers.append(sampler)
                loader = ParallelDataLoader(dls)
                return loader, samplers
            else:
                loader, sampler = get_single_loader(dss, split_conf.loader_params)
                return loader, [sampler]

        train_dl, train_samplers = get_loader(train_ds, self.conf.train)
        if self.do_val:
            val_dl, val_samplers = get_loader(val_ds, self.conf.val)
        else:
            val_dl, val_samplers = None, [None]
        if self.do_test:
            test_dl, test_samplers = get_loader(test_ds, self.conf.test)
        else:
            test_dl, test_samplers = None, [None]

        # validate the dataloaders
        if len(train_dl) == 0:
            raise AttributeError("'train' dataloader has length 0")
        if val_dl is not None and (len(val_dl) == 0):
            raise AttributeError("'val' dataloader has length 0")
        if test_dl is not None and (len(test_dl) == 0):
            raise AttributeError("'test' dataloader has length 0")

        return (
            (train_dl, val_dl, test_dl),
            (*train_samplers, *val_samplers, *test_samplers),
        )

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
        train_batch_count = len(train_dl)

        if self.do_val:
            val_batch_count = len(val_dl)
        else:
            val_batch_count = None

        dl_keys = None  # TODO: get rid of this
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
            # for smpl_l1 in samplers:
            #     if type(smpl_l1) == dict:
            #         for smpl_l2 in smpl_l1.values():
            #             if smpl_l2 is not None:
            #                 smpl_l2.set_epoch(epoch)
            #     else:
            #         if smpl_l1 is not None:
            #             smpl_l1.set_epoch(epoch)
            for smpl in samplers:
                if smpl is not None:
                    smpl.set_epoch(epoch)

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
