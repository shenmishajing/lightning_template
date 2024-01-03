from copy import deepcopy

from lightning.pytorch.cli import instantiate_class
from lightning.pytorch.core.datamodule import (
    LightningDataModule as _LightningDataModule,
)
from sklearn.model_selection import KFold
from torch.utils.data import (
    DataLoader,
    IterableDataset,
    IterDataPipe,
    RandomSampler,
    SequentialSampler,
    Subset,
)
from torch.utils.data.dataloader import _InfiniteConstantSampler
from torch.utils.data.graph_settings import apply_shuffle_settings

from lightning_template.utils.cli import recursive_instantate_class
from lightning_template.utils.mixin import SplitNameMixin


class LightningDataModule(SplitNameMixin, _LightningDataModule):
    def __init__(
        self,
        dataset_cfg: dict = None,
        dataloader_cfg: dict = None,
    ):
        super().__init__()
        self.datasets = {}
        self.dataset = None
        self.num_folds = None
        self.folds = {}
        self.splits = []
        self.batch_size = None

        self.dataset_cfg = self.get_split_config(dataset_cfg)
        self.dataloader_cfg = self.get_split_config(dataloader_cfg)

    def build_dataset(self, split):
        self.datasets[split] = recursive_instantate_class(self.dataset_cfg[split])

    def build_collate_fn(self, collate_fn_cfg, dataset):
        if hasattr(dataset, "collate_fn"):
            return dataset.collate_fn
        elif collate_fn_cfg:
            return recursive_instantate_class(collate_fn_cfg)
        else:
            return None

    def build_sampler(self, dataloader_cfg, dataset, split):
        if isinstance(dataset, IterableDataset):
            shuffle = dataloader_cfg.pop("shuffle", None)
            if isinstance(dataset, IterDataPipe) and shuffle is not None:
                dataset = apply_shuffle_settings(dataset, shuffle=shuffle)
            return _InfiniteConstantSampler()
        else:
            if dataloader_cfg.pop("shuffle", split == self.TrainSplit):
                return RandomSampler(dataset)
            else:
                return SequentialSampler(dataset)

    def build_batch_sampler(self, batch_sampler_cfg, dataset, *args):
        return instantiate_class(args, batch_sampler_cfg)

    def handle_dataloader_config(self, dataloader_cfg, dataset, split, *arg, **kwargs):
        if "batch_sampler" in dataloader_cfg:
            if "init_args" not in dataloader_cfg["batch_sampler"]:
                dataloader_cfg["batch_sampler"]["init_args"] = {}

            dataloader_cfg["batch_sampler"]["init_args"][
                "batch_size"
            ] = dataloader_cfg.pop("batch_size", 1)
            dataloader_cfg["batch_sampler"]["init_args"][
                "drop_last"
            ] = dataloader_cfg.pop("drop_last", False)
            dataloader_cfg["batch_sampler"]["init_args"][
                "sampler"
            ] = self.build_sampler(dataloader_cfg, dataset, split)

            dataloader_cfg["batch_sampler"] = self.build_batch_sampler(
                dataloader_cfg["batch_sampler"], dataset
            )
        elif "sampler" not in dataloader_cfg and not isinstance(
            dataset, IterableDataset
        ):
            dataloader_cfg.setdefault("shuffle", split == self.TrainSplit)

        return dataloader_cfg

    def _build_dataloader(self, dataset, dataloader_cfg, split):
        set_batch_size = split == self.split_names[0]
        if set_batch_size:
            dataloader_cfg["batch_size"] = self.batch_size
        dataloader_cfg["collate_fn"] = self.build_collate_fn(
            dataloader_cfg.get("collate_fn", {}), dataset
        )
        return DataLoader(
            dataset,
            **self.handle_dataloader_config(dataloader_cfg, dataset, split=split),
        )

    def build_dataloader(self, split):
        dataset = self.datasets[split] if self.num_folds is None else self.folds[split]
        dataloader_cfg = self.dataloader_cfg.get(split, {})

        if isinstance(dataset, list):
            result = []
            if not isinstance(dataloader_cfg, list):
                dataloader_cfg = [deepcopy(dataloader_cfg) for _ in range(len(dataset))]
            for d, cfg in zip(dataset, dataloader_cfg):
                result.append(self._build_dataloader(d, cfg, split))
        elif isinstance(dataset, dict):
            result = {}
            if not all([k in dataloader_cfg for k in dataset]):
                dataloader_cfg = {k: deepcopy(dataloader_cfg) for k in dataset}
            for k in dataset:
                result[k] = self._build_dataloader(dataset[k], dataloader_cfg[k], split)
        else:
            result = self._build_dataloader(dataset, deepcopy(dataloader_cfg), split)

        return result

    def setup(self, stage=None):
        super().setup(stage=stage)

        for name in self.split_names:
            self.build_dataset(name)
        self.dataset = self.datasets[self.split_names[0]]
        self.batch_size = self.dataloader_cfg[self.split_names[0]].get("batch_size", 1)

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [
            split for split in KFold(num_folds).split(range(len(self.dataset)))
        ]

    def setup_fold_index(self, fold_index: int) -> None:
        for indices, fold_name in zip(
            self.splits[fold_index], [self.TrainSplit, self.ValidateSplit]
        ):
            self.folds[fold_name] = Subset(self.dataset, indices)

        for fold_name in [self.TestSplit, self.PredictSplit]:
            self.folds[fold_name] = self.folds[self.ValidateSplit]

    def train_dataloader(self):
        return self.build_dataloader(self.TrainSplit)

    def val_dataloader(self):
        return self.build_dataloader(self.ValidateSplit)

    def test_dataloader(self):
        return self.build_dataloader(self.TestSplit)

    def predict_dataloader(self):
        return self.build_dataloader(self.PredictSplit)
