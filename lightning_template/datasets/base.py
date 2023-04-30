import copy
import string
from collections.abc import Mapping
from typing import List

from lightning.pytorch.cli import instantiate_class
from lightning.pytorch.core.datamodule import (
    LightningDataModule as _LightningDataModule,
)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset

from lightning_template.utils import deep_update


class LightningDataModule(_LightningDataModule):
    def __init__(
        self,
        dataset_cfg: dict = None,
        dataloader_cfg: dict = None,
    ):
        super().__init__()
        self.split_names = ["train", "val", "test", "predict"]
        self.datasets = {}
        self.dataset = None
        self.num_folds = None
        self.folds = {}
        self.splits = []
        self.batch_size = None

        self.dataset_cfg = self.get_split_config(dataset_cfg)
        self.dataloader_cfg = self.get_split_config(dataloader_cfg)

    def get_split_config(self, config):
        if isinstance(config, Mapping):
            if all([config.get(name) is None for name in self.split_names]):
                return {name: copy.deepcopy(config) for name in self.split_names}
            else:
                res = {}
                last_name = None
                for name in self.split_names:
                    if last_name is None:
                        res[name] = copy.deepcopy(config[name])
                    else:
                        res[name] = deep_update(
                            copy.deepcopy(res[last_name]), config.get(name, {})
                        )
                    last_name = name

                if "split_info" in config and "split_format_to" in config["split_info"]:
                    config = config["split_info"]

                    if not isinstance(config["split_format_to"], List):
                        config["split_format_to"] = [config["split_format_to"]]

                    split_name_map = {
                        "train": "train",
                        "val": "val",
                        "test": "val",
                        "predict": "val",
                    }
                    split_name_map.update(config.get("split_name_map", {}))
                    config["split_name_map"] = split_name_map

                    config.setdefault("split_prefix", "init_args")
                    config.setdefault("split_attr_split_str", ".")

                    for name in self.split_names:
                        for split_attr in config["split_format_to"]:
                            cur_cfg = res[name]
                            if config["split_prefix"] is not None:
                                for s in config["split_prefix"].split(
                                    config["split_attr_split_str"]
                                ):
                                    if s not in cur_cfg:
                                        cur_cfg[s] = {}
                                    cur_cfg = cur_cfg[s]

                            split_attr = split_attr.split(
                                config["split_attr_split_str"]
                            )
                            for s in split_attr[:-1]:
                                cur_cfg = cur_cfg[s]
                            split_attr = split_attr[-1]
                            cur_cfg[split_attr] = string.Template(
                                cur_cfg.get(split_attr, "$split")
                            ).safe_substitute(split=config["split_name_map"][name])
                return res
        else:
            return {
                name: copy.deepcopy(config) if config else {}
                for name in self.split_names
            }

    def _build_dataset(self, split):
        self.datasets[split] = instantiate_class(tuple(), self.dataset_cfg[split])

    def _build_collate_fn(self, collate_fn_cfg):
        return None

    def _build_sampler(self, dataloader_cfg, dataset):
        if "shuffle" in dataloader_cfg:
            shuffle = dataloader_cfg.pop("shuffle")
        else:
            shuffle = False

        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        return sampler

    def _build_batch_sampler(self, batch_sampler_cfg, dataset, *args):
        return instantiate_class(args, batch_sampler_cfg)

    def _handle_batch_sampler(self, dataloader_cfg, dataset, split="train"):
        if "batch_sampler" in dataloader_cfg:
            dataloader_cfg["batch_sampler"] = self._build_batch_sampler(
                dataloader_cfg["batch_sampler"],
                dataset,
                self._build_sampler(dataloader_cfg, dataset),
                dataloader_cfg.pop("batch_size", 1),
                dataloader_cfg.pop("drop_last", False),
            )
        return dataloader_cfg

    def _build_dataloader(self, dataset, split="train", set_batch_size=False):
        dataloader_cfg = copy.deepcopy(self.dataloader_cfg.get(split, {}))
        if set_batch_size:
            dataloader_cfg["batch_size"] = self.batch_size
        dataloader_cfg["collate_fn"] = self._build_collate_fn(
            dataloader_cfg.get("collate_fn", {})
        )
        return DataLoader(
            dataset, **self._handle_batch_sampler(dataloader_cfg, dataset, split=split)
        )

    def _dataloader(self, split, **kwargs):
        return self._build_dataloader(
            self.datasets[split] if self.num_folds is None else self.folds[split],
            split=split,
            set_batch_size=split == self.split_names[0],
            **kwargs
        )

    def _get_split_names(self, stage=None):
        if self.trainer.overfit_batches > 0:
            split_names = ["train"]
        elif stage is None:
            split_names = self.split_names
        elif stage == "fit":
            split_names = ["train", "val"]
        elif stage == "validate":
            split_names = ["val"]
        else:
            split_names = [stage.lower()]
        return split_names

    def setup(self, stage=None):
        self.split_names = self._get_split_names(stage)

        for name in self.split_names:
            self._build_dataset(name)
        self.dataset = self.datasets[self.split_names[0]]
        self.batch_size = self.dataloader_cfg[self.split_names[0]].get("batch_size", 1)

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [
            split for split in KFold(num_folds).split(range(len(self.dataset)))
        ]

    def setup_fold_index(self, fold_index: int) -> None:
        for indices, fold_name in zip(self.splits[fold_index], ["train", "val"]):
            self.folds[fold_name] = Subset(self.dataset, indices)

        for fold_name in ["test", "predict"]:
            self.folds[fold_name] = self.folds["val"]

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def test_dataloader(self):
        return self._dataloader("test")

    def predict_dataloader(self):
        return self._dataloader("predict")
