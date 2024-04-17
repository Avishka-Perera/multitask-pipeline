import torch
import os, shutil
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from typing import Sequence, Dict, Tuple, Any
import yaml
from torch.utils.data import Subset, DataLoader
from torch import autocast
from tqdm import tqdm
import pandas as pd
import omegaconf
from ..logger import Logger
from ..util import (
    yaml_loader,
    load_class,
    make_obj_from_conf,
    fix_list_len,
    load_config,
    load_model_states,
    are_lists_equal,
)
from ..dist import get_is_dist, get_mixed_prec
from ..reporting import history_to_csv, history_to_img
from ...losses import BaseLoss
from ...evaluators import BaseEvaluator
from ...visualizers import BaseVisualizer
from ...learners import BaseLearner
from ...datasets import BaseDataset, ConcatSet
from ..data import ParallelDataLoader
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
from functools import partial, reduce
from torch.utils.data._utils.collate import default_collate
import datetime, time
import glob
from .util import validate_conf, load_datasets


class Trainer:
    already_trained_msg = "Training already done!"

    def _load_datasets(self) -> None:
        self.datasets = load_datasets(self.conf)

    def _load_learner(self):
        learner_class = load_class(self.conf.learner.target)

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

        if "params" in self.conf.learner.keys():
            learner: BaseLearner = learner_class(**dict(self.conf.learner.params))
        else:
            learner: BaseLearner = learner_class()

        # map the devices
        if "local_device_maps" in self.conf.learner.keys():

            def map_devices(local_device_maps, device_count):
                devices = [self.devices[local_id] for local_id in local_device_maps]
                # correct the devices
                devices = get_correct_device_lst(devices, device_count)
                return devices

            if type(self.conf.learner.local_device_maps) == ListConfig:
                devices = map_devices(
                    self.conf.learner.local_device_maps, learner_class.device_count
                )
            else:
                if (
                    self.conf.learner.target
                    != "mt_pipe.src.util.learner_mux.LearnerMux"
                ):
                    raise ValueError(
                        "If not using 'mt_pipe.src.util.learner_mux.LearnerMux', local_device_maps must be an integer list"
                    )
                if not are_lists_equal(
                    list(self.conf.learner.local_device_maps.keys()),
                    list(self.conf.learner.params.chldrn.keys()),
                ):
                    raise ValueError(
                        "When local_device_maps are specified, 'conf.learner.local_device_maps' and 'conf.learner.params.chldrn' must have the same keys."
                    )

                devices = {}
                for k in self.conf.learner.local_device_maps.keys():
                    ch_ln_cls = load_class(self.conf.learner.params.chldrn[k].target)
                    device_count = ch_ln_cls.device_count
                    ch_devices = map_devices(
                        self.conf.learner.local_device_maps[k], device_count
                    )
                    devices[k] = ch_devices
        else:
            devices = self.devices
            devices = get_correct_device_lst(devices, learner_class.device_count)
        self.logger.info(f"Learner: using devices: {devices}")

        learner.set_devices(devices)
        if self.is_dist:
            torch.cuda.set_device(devices[0])
            self.learner = DDP(learner)
        else:
            self.learner = learner

    def _load_states(self, ckpt_path: str, ckpt_map_conf_path: str = None) -> None:
        ckpt = torch.load(ckpt_path)
        if ckpt_map_conf_path is not None:
            with open(ckpt_map_conf_path) as handler:
                ckpt_map_info = OmegaConf.create(yaml.load(handler, yaml.FullLoader))
        else:
            ckpt_map_info = None

        # load learner
        learner = self.learner.module if self.is_dist else self.learner
        if "learner" in ckpt:
            sd = ckpt["learner"]
        else:
            self.logger.warn(
                "'learner' key not found in the provided checkpoint. Assuming checkpoint is a state dict"
            )
            sd = ckpt
        if ckpt_map_conf_path is None or "learner" not in ckpt_map_info:
            if hasattr(learner, "load_ckeckpoint"):
                learner.load_ckeckpoint(ckpt_path)
            else:
                learner.load_state_dict(sd)
        else:
            learner_map_info = ckpt_map_info.learner
            load_model_states(learner, sd, learner_map_info)

    def _load_losses(self):
        if "losses" in self.conf:
            self.losses = {}
            for loss_nm, loss_conf in self.conf.losses.items():
                # map devices
                if "local_device_map" in loss_conf.keys():
                    device = self.devices[loss_conf.local_device_map]
                else:
                    device = self.devices[-1]
                self.losses[loss_nm] = make_obj_from_conf(loss_conf, device=device)

    def _load_training_objects(self):
        if self.do_train:
            if "optimizer" in self.conf:
                optim_class = load_class(self.conf.optimizer.target)
                params = (
                    dict(self.conf.optimizer.params)
                    if "params" in self.conf.optimizer
                    else {}
                )
                self.optimizer = optim_class(self.learner.parameters(), **params)
            else:
                self.optimizer = Adam(self.learner.parameters())
            self.logger.info(f"Using optimizer {self.optimizer.__class__.__name__}")

            if "lr_scheduler" in self.conf:
                scheduler_class = load_class(self.conf.lr_scheduler.target)
                params = (
                    self.conf.lr_scheduler.params
                    if "params" in self.conf.lr_scheduler
                    else {}
                )
                self.lr_scheduler: LRScheduler = scheduler_class(
                    self.optimizer, **params
                )
                self.logger.info(
                    f"Using scheduler {self.lr_scheduler.__class__.__name__}"
                )
            else:
                self.lr_scheduler = None
                self.logger.info("Not using any scheduler")

            if self.use_amp:
                self.mp_dtype, self.scaler = get_mixed_prec()
                self.logger.info(f"Using AMP with dtype {self.mp_dtype}")

    def _validate_conf(self):
        validate_conf(
            conf=self.conf,
            data_dir=self.data_dir,
            devices=self.devices,
            logger=self.logger,
        )

    def _load_evaluator(self) -> None:
        self.evaluators: Dict[str, BaseEvaluator] = {}
        if self.do_test:
            for eval_nm, eval_conf in self.conf.test.evaluators.items():
                eval_class = load_class(eval_conf.target)
                params = dict(eval_conf.params) if "params" in eval_conf else {}
                self.evaluators[eval_nm] = eval_class(**params)

    def _load_visualizer(self) -> None:
        self.visualizers: Dict[str, BaseVisualizer] = {}
        if "visualizers" in self.conf:
            for visu_nm, visu_conf in self.conf.visualizers.items():
                self.visualizers[visu_nm] = make_obj_from_conf(visu_conf, name=visu_nm)

    def _load_augmentors(self) -> None:
        self.augmentors = {}
        if "augmentors" in self.conf:
            for aug_nm, aug_conf in self.conf.augmentors.items():
                self.augmentors[aug_nm] = make_obj_from_conf(aug_conf)

    def __init__(
        self,
        conf: str | Dict | omegaconf.dictconfig.DictConfig,
        data_dir: str,
        weights_conf: Dict[str, str],
        devices: Sequence[str | int],
        use_amp: bool,
        logger: Logger,
        analysis_level: int,
        visualize_every: int,
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
        self.data_dir = data_dir
        self.devices = devices

        is_dist = get_is_dist()
        if is_dist:
            self.is_dist = True
            self.rank, self.local_rank, self.world_size = is_dist
            self.do_out = self.rank == 0
        else:
            self.is_dist = False
            self.rank, self.local_rank, self.world_size = 0, 0, 1
            self.do_out = True

        self.logger = Logger(0) if logger is None else logger
        self._validate_conf()
        self.do_train = "train" in self.conf.keys()
        self.do_val = "val" in self.conf.keys()
        self.do_test = "test" in self.conf.keys()
        self.iteration = 0
        if not self.do_val:
            self.logger.warn(
                "No validation configuration detected. Validation loops will be skipped"
            )
        self._load_datasets()
        self._load_learner()
        if weights_conf["ckpt_path"] is not None:
            self._load_states(
                weights_conf["ckpt_path"], weights_conf["ckpt_map_conf_path"]
            )
        self._load_losses()
        self._load_evaluator()
        self._load_visualizer()
        self._load_augmentors()
        self.use_amp = use_amp
        self._load_training_objects()
        self.ckpts = (
            {ckpt.epoch: ckpt.name for ckpt in self.conf.checkpoints}
            if "checkpoints" in self.conf.keys()
            else {}
        )
        self.analysis_level = analysis_level
        self.visualize_every = visualize_every

    def val_step(
        self,
        batch: Sequence[torch.Tensor] | Dict[str, Sequence[torch.Tensor]],
        epoch: int,
        batch_id: int,
        batch_count: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch["curr_epoch"] = epoch
        if self.use_amp:
            with autocast(device_type="cuda", dtype=self.mp_dtype):
                info = self.learner(batch)
                loss_pack = self.val_loss_fn(info=info, batch=batch)
        else:
            info = self.learner(batch)
            loss_pack = self.val_loss_fn(info=info, batch=batch)
        info = self.learner(batch)
        loss_pack = self.val_loss_fn(info=info, batch=batch)
        tot_loss = loss_pack["tot"]

        if self.analysis_level > 0 and self.do_out:
            self.logger.plot_loss_pack(loss_pack, epoch * batch_count + batch_id, "val")

        return tot_loss, info

    def train_step(
        self,
        batch: Sequence[torch.Tensor] | Dict[str, Sequence[torch.Tensor]],
        epoch: int,
        batch_id: int,
        batch_count: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        self.optimizer.zero_grad()
        batch["curr_epoch"] = epoch

        if self.use_amp:
            with autocast(device_type="cuda", dtype=self.mp_dtype):
                info = self.learner(batch)
                loss_pack = self.train_loss_fn(info=info, batch=batch)
            tot_loss = loss_pack["tot"]
            self.scaler.scale(tot_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            info = self.learner(batch)
            loss_pack = self.train_loss_fn(info=info, batch=batch)
            tot_loss = loss_pack["tot"]
            tot_loss.backward()
            self.optimizer.step()

        if self.lr_scheduler:
            self.lr_scheduler.step(epoch + (batch_id + 1) / batch_count)

        if self.analysis_level > 0 and self.do_out:
            self.logger.batch_step(analyze_grads=self.analysis_level > 1)
            self.logger.plot_loss_pack(
                loss_pack, epoch * batch_count + batch_id, "train"
            )

        return tot_loss, info

    def _save_ckpt(self, epoch, output_path, status, history=None):
        # checkpoints
        ckpt_path = os.path.join(output_path, "ckpts", status + ".ckpt")
        learner_state = (
            self.learner.module.state_dict()
            if self.is_dist
            else self.learner.state_dict()
        )
        optim_state = self.optimizer.state_dict()
        lr_scheduler_state = (
            None if self.lr_scheduler is None else self.lr_scheduler.state_dict()
        )
        ckpts = {
            "learner": learner_state,
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
        learner_state = ckpts["learner"]
        optim_state = ckpts["optimizer"]
        lr_scheduler_state = ckpts["lr_scheduler"]
        epoch = ckpts["epoch"]
        if self.is_dist:
            self.learner.module.load_state_dict(learner_state)
        else:
            self.learner.load_state_dict(learner_state)
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

        if self.is_dist:
            dist.barrier()

        if self.do_out:
            os.makedirs(os.path.join(output_path, "ckpts"))

        return output_path

    def _init_fit(
        self,
        args,
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

            # find the latest resumption
            old_config_paths = glob.glob(
                os.path.join(resume_dir, "configuration*.yaml")
            )
            resume_id = len(old_config_paths)
            if resume_id == 1:
                old_config_path = os.path.join(resume_dir, "configuration.yaml")
                assert old_config_paths[0] == old_config_path
            else:
                old_config_path = os.path.join(
                    resume_dir, f"configuration{resume_id-1}.yaml"
                )
            new_config_path = os.path.join(resume_dir, f"configuration{resume_id}.yaml")
            new_args_path = os.path.join(resume_dir, f"args{resume_id}.yaml")

            old_config = load_config(old_config_path)
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

                if "val_loss" in history:
                    min_loss = history["val_loss"].min()
                    best_epoch = history["val_loss"].argmin()
                else:
                    min_loss = float("inf")
                    best_epoch = -1

                history = history.to_dict()
                del history["epoch"]
                history["train_loss"] = list(history["train_loss"].values())

                if "val_loss" in history:
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
            model = self.learner.module if self.is_dist else self.learner
            self.logger.init_plotter(os.path.join(output_path, "logs"), model)
            for vis in self.visualizers.values():
                vis.set_writer(self.logger.writer)
            if self.do_test:
                for nm, eval in self.evaluators.items():
                    eval.set_out_path(os.path.join(output_path, "evals", nm))

            # log the invocation information
            info = [
                "Start time: "
                + datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S " + f"({time.tzname[0]})"
                ),
            ]
            info_path = os.path.join(output_path, "info.txt")
            if resume_dir is None:
                with open(info_path, "a") as handler:
                    handler.write("".join(info))

                with open(os.path.join(output_path, "configuration.yaml"), "w") as file:
                    yaml.dump(OmegaConf.to_container(self.conf), file)
                with open(os.path.join(output_path, "args.yaml"), "w") as file:
                    yaml.dump(vars(args), file)
            else:
                info = ["\n", f"----------- Resumption {resume_id} -----------"] + info
                with open(info_path, "a") as handler:
                    handler.write("\n".join(info))

                with open(new_config_path, "w") as file:
                    yaml.dump(OmegaConf.to_container(self.conf), file)
                with open(new_args_path, "w") as file:
                    yaml.dump(vars(args), file)

        return output_path, min_loss, start_epoch, best_epoch, history

    def _get_datasets(
        self, mock_batch_count: Sequence[int]
    ) -> Sequence[Dataset] | Sequence[Dict[str, Dataset]]:
        """
        Collects the loaded datasets according to the requests in the loop definitions.
            If a loop definition requests a single dataset (e.g.: `conf.train.dataset: ds-1`), then a usual dataset object is returned
            If a loop definition requests multiple datasets (e.g.: `conf.train.dataset: {task1: ds-1, task2: ds-2}`), then a distionary containing the requested datasets is returned
                This is used in the case of multi-task training with parallel dataloaders

        Furthermore, the length of the datasets are adjusted according to the specified mock_bacth_count
            default is -1. i.e., dataset is collected as it is
            If mock_batch_count is specified, the dataset is wrapped in a Subset object to provide the requested mock_batch_count
            mock_batch_count is a list with either one integer or three integers
                in the case of one integer, all the loops will iterate over a same number of batches
                in the case of multiple integers, three loops will iterate over different number of batches as defined
        """
        assert len(mock_batch_count) in [1, 3]
        train_mock_batch_count = mock_batch_count[0]
        if len(mock_batch_count) == 3:
            val_mock_batch_count = mock_batch_count[1]
            test_mock_batch_count = mock_batch_count[2]
        else:
            val_mock_batch_count = train_mock_batch_count
            test_mock_batch_count = train_mock_batch_count

        def wrap_dataset(dataset, mock_batch_count, batch_size):
            if mock_batch_count == -1:
                return dataset
            else:
                return Subset(
                    dataset,
                    range(
                        min(
                            batch_size * mock_batch_count * self.world_size,
                            len(dataset),
                        )
                    ),
                )

        train_ds, val_ds, test_ds = [None] * 3
        if self.do_train:
            if type(self.conf.train.dataset) == str:
                train_ds = wrap_dataset(
                    self.datasets[self.conf.train.dataset],
                    train_mock_batch_count,
                    self.conf.train.loader_params.batch_size,
                )
            else:
                train_ds = {
                    k: wrap_dataset(
                        self.datasets[v],
                        train_mock_batch_count,
                        self.conf.train.loader_params.batch_size,
                    )
                    for k, v in self.conf.train.dataset.items()
                }
        if self.do_val:
            if type(self.conf.val.dataset) == str:
                val_ds = wrap_dataset(
                    self.datasets[self.conf.val.dataset],
                    val_mock_batch_count,
                    self.conf.val.loader_params.batch_size,
                )
            else:
                val_ds = {
                    k: wrap_dataset(
                        self.datasets[v],
                        val_mock_batch_count,
                        self.conf.val.loader_params.batch_size,
                    )
                    for k, v in self.conf.val.dataset.items()
                }
        if self.do_test:
            if type(self.conf.test.dataset) == str:
                test_ds = wrap_dataset(
                    self.datasets[self.conf.test.dataset],
                    test_mock_batch_count,
                    self.conf.test.loader_params.batch_size,
                )
            else:
                test_ds = {
                    k: wrap_dataset(
                        self.datasets[v],
                        test_mock_batch_count,
                        self.conf.test.loader_params.batch_size,
                    )
                    for k, v in self.conf.test.dataset.items()
                }

        return train_ds, val_ds, test_ds

    def _get_dataloaders(
        self,
        train_ds: Sequence[Dataset] | Sequence[Dict[str, Dataset]],
        val_ds: Sequence[Dataset] | Sequence[Dict[str, Dataset]],
        test_ds: Sequence[Dataset] | Sequence[Dict[str, Dataset]],
    ):

        def process_loader_params(split_conf: OmegaConf, ds) -> Dict:
            new_loader_params = OmegaConf.to_container(split_conf.loader_params)

            # add the augmentor
            if "augmentor" in split_conf:
                augmentor = self.augmentors[split_conf.augmentor]

                def new_collate_fn(batch):
                    if hasattr(augmentor, "pre_collate_routine"):
                        batch = [
                            augmentor.pre_collate_routine(sample) for sample in batch
                        ]
                    batch = default_collate(batch)
                    if hasattr(augmentor, "post_collate_routine"):
                        batch = augmentor.post_collate_routine(batch)
                    if not (
                        hasattr(augmentor, "pre_collate_routine")
                        or hasattr(augmentor, "post_collate_routine")
                    ):
                        batch = augmentor(batch)
                    return batch

                new_loader_params["collate_fn"] = new_collate_fn

            if self.is_dist:
                if "sampler" in split_conf.loader_params:
                    sampler_class = load_class(split_conf.loader_params.sampler.target)
                else:
                    sampler_class = DistributedSampler
                new_loader_params["sampler"] = sampler_class(
                    ds, self.world_size, self.rank
                )
            else:
                if "sampler" in split_conf.loader_params:
                    raise NotImplementedError(
                        "Custom sampler usage is not implemented for non-DDP setups"
                    )

            return new_loader_params

        def get_loader(dss, split_conf):
            def get_single_loader(ds, split_conf):
                params = process_loader_params(split_conf, ds)
                sampler = params["sampler"] if "sampler" in params else None
                loader = DataLoader(ds, **dict(params))
                return loader, sampler

            if type(dss) == dict:
                samplers, dls = [], []
                for nm, ds in dss.items():
                    loader, sampler = get_single_loader(ds, split_conf[nm])
                    dls.append(loader)
                    samplers.append(sampler)
                loader = ParallelDataLoader(dls)
                return loader, samplers
            else:
                loader, sampler = get_single_loader(dss, split_conf)
                return loader, [sampler]

        if self.do_train:
            train_dl, train_samplers = get_loader(train_ds, self.conf.train)
        else:
            train_dl, train_samplers = None, [None]
        if self.do_val:
            val_dl, val_samplers = get_loader(val_ds, self.conf.val)
        else:
            val_dl, val_samplers = None, [None]
        if self.do_test:
            test_dl, test_samplers = get_loader(test_ds, self.conf.test)
        else:
            test_dl, test_samplers = None, [None]

        # validate the dataloaders
        if train_dl is not None and (len(train_dl) == 0):
            raise AttributeError("'train' dataloader has length 0")
        if val_dl is not None and (len(val_dl) == 0):
            raise AttributeError("'val' dataloader has length 0")
        if test_dl is not None and (len(test_dl) == 0):
            raise AttributeError("'test' dataloader has length 0")

        return (
            (train_dl, val_dl, test_dl),
            (*train_samplers, *val_samplers, *test_samplers),
        )

    def _train_loop(self, train_batch_count, show_pbar, epoch, train_dl):
        # train loop
        self.learner.train()
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
                train_loss, info = self.train_step(
                    batch, epoch, batch_id, train_batch_count
                )
                train_loss = train_loss.detach().cpu().item()
                if self.do_out:
                    pbar.set_postfix(loss=train_loss)
                    pbar.update(1)
                    if (
                        (batch_id % self.visualize_every == 0)
                        or (batch_id == train_batch_count - 1)
                    ) and "visualizer" in self.conf.train:
                        self.visualizers[self.conf.train.visualizer](
                            info, batch, epoch, "train"
                        )
                train_losses.append(train_loss)

            for batch_id, batch in enumerate(train_dl):
                process_batch(batch, batch_id)

        train_loss = sum(train_losses) / len(train_losses)

        return train_loss

    def _val_loop(self, val_batch_count, show_pbar, epoch, val_dl):
        self.learner.eval()
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
                    val_loss_step, info = self.val_step(
                        batch, epoch, batch_id, val_batch_count
                    )
                    val_loss += val_loss_step
                    cur_val_loss = val_loss / (batch_id + 1)
                    if self.do_out:
                        pbar.set_postfix(loss=cur_val_loss.detach().cpu().item())
                        pbar.update(1)
                        if (
                            (batch_id % self.visualize_every == 0)
                            or (batch_id == val_batch_count - 1)
                        ) and "visualizer" in self.conf.val:
                            self.visualizers[self.conf.val.visualizer](
                                info, batch, epoch, "val"
                            )

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

        if self.is_dist:
            dist.all_reduce(val_loss, dist.ReduceOp.SUM)
            val_loss = val_loss / self.world_size
        val_loss = val_loss.detach().cpu().item()

        return val_loss

    def _test_loop(self, show_pbar: bool, epoch: int, test_dl: DataLoader) -> None:
        self.learner.eval()
        with torch.no_grad():
            with tqdm(
                total=len(test_dl),
                disable=not (self.logger.display_info and show_pbar and self.do_out),
                desc="Testing",
            ) as pbar:
                if not show_pbar:
                    self.logger.info("Testing")

                results = {nm: [] for nm in self.evaluators.keys()}
                for batch_id, batch in enumerate(test_dl):
                    batch["curr_epoch"] = epoch
                    if self.use_amp:
                        with autocast(device_type="cuda", dtype=self.mp_dtype):
                            info = self.learner(batch)
                    else:
                        info = self.learner(batch)

                    for nm, eval in self.evaluators.items():
                        results[nm].append(eval.process_batch(batch=batch, info=info))
                    pbar.update()
                    if (
                        (batch_id % self.visualize_every == 0)
                        or (batch_id == len(test_dl) - 1)
                    ) and "visualizer" in self.conf.test:
                        self.visualizers[self.conf.test.visualizer](
                            info, batch, epoch, "test"
                        )

                # gather all results at rank 0 replica
                if self.is_dist:
                    all_results = [None for _ in range(self.world_size)]

                    if self.rank == 0:
                        dist.gather_object(results, all_results)
                    else:
                        dist.gather_object(results)

                    if self.rank == 0:
                        results = reduce(
                            lambda i, res: {k: v + res[k] for k, v in i.items()},
                            all_results,
                            {k: [] for k in all_results[0].keys()},
                        )

                # output the results
                if self.do_out:
                    for nm, eval in self.evaluators.items():
                        eval.output(results[nm])

    def _get_fit_info(self, mock_epoch_count, train_dl, val_dl):
        tollerance = (
            self.conf.train.tollerance
            if "train" in self.conf and "tollerance" in self.conf.train
            else -1
        )
        if self.do_train:
            train_batch_count = len(train_dl)
        else:
            train_batch_count = None

        if self.do_val:
            val_batch_count = len(val_dl)
        else:
            val_batch_count = None

        if self.do_train:
            epochs = mock_epoch_count if mock_epoch_count > 0 else self.conf.epochs
        else:
            epochs = 0

        return (
            tollerance,
            train_batch_count,
            val_batch_count,
            train_dl,
            val_dl,
            epochs,
        )

    def fit(
        self,
        args,
        output_path: str = "out",
        run_name: str = None,
        mock_batch_count: Sequence[int] = [-1],
        mock_epoch_count: int = -1,
        resume_dir: str = None,
        force_resume: bool = False,
        show_pbar=True,
    ) -> None:

        output_path, min_loss, start_epoch, best_epoch, history = self._init_fit(
            args, output_path, run_name, resume_dir, force_resume
        )

        train_ds, val_ds, test_ds = self._get_datasets(mock_batch_count)
        loaders, samplers = self._get_dataloaders(train_ds, val_ds, test_ds)
        train_dl, val_dl, test_dl = loaders

        (
            tollerance,
            train_batch_count,
            val_batch_count,
            train_dl,
            val_dl,
            epochs,
        ) = self._get_fit_info(mock_epoch_count, train_dl, val_dl)

        if start_epoch >= epochs:
            self.logger.info(self.already_trained_msg)

        best_ckpt_path = os.path.join(output_path, "ckpts", "best.ckpt")
        final_ckpt_path = os.path.join(output_path, "ckpts", "final.ckpt")
        self.logger.info(f"Saving outputs to {output_path}")

        def set_samplers_epoch(epoch):
            for smpl in samplers:
                if smpl is not None:
                    smpl.set_epoch(epoch)

        if self.do_train:
            for epoch in range(start_epoch, epochs):
                self.logger.info(
                    f"---- EPOCH {str(epoch+1).rjust(len(str(epochs)), '0')}/{epochs} ----"
                )
                set_samplers_epoch(epoch)

                train_loss = self._train_loop(
                    train_batch_count, show_pbar, epoch, train_dl
                )
                torch.cuda.empty_cache()

                if self.do_val:
                    val_loss = self._val_loop(val_batch_count, show_pbar, epoch, val_dl)
                    torch.cuda.empty_cache()
                else:
                    val_loss = None

                if self.is_dist:
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

                torch.cuda.empty_cache()
                gc.collect()
                gc.collect(0)
                gc.collect(1)
                gc.collect(2)
                torch.cuda.empty_cache()

            if self.is_dist:
                dist.barrier()

        if self.do_test:
            if self.do_train:
                if os.path.exists(best_ckpt_path):
                    sd = torch.load(best_ckpt_path)["learner"]
                else:
                    sd = torch.load(final_ckpt_path)["learner"]
                if self.is_dist:
                    self.learner.module.load_state_dict(sd)
                else:
                    self.learner.load_state_dict(sd)

            self._test_loop(show_pbar=show_pbar, epoch=epoch, test_dl=test_dl)

        self.logger.info(f"Single-staged training successful!\n")
