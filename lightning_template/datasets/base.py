import copy

from lightning.pytorch.cli import instantiate_class
from lightning.pytorch.core.datamodule import (
    LightningDataModule as _LightningDataModule,
)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset

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

    def _build_dataset(self, split):
        self.datasets[split] = recursive_instantate_class(self.dataset_cfg[split])

    def _build_collate_fn(self, collate_fn_cfg, dataset):
        if hasattr(dataset, "collate_fn"):
            return dataset.collate_fn
        elif collate_fn_cfg:
            return recursive_instantate_class(collate_fn_cfg)
        else:
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

    def _handle_batch_sampler(self, dataloader_cfg, dataset, *arg, **kwargs):
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
            ] = self._build_sampler(dataloader_cfg, dataset)

            dataloader_cfg["batch_sampler"] = self._build_batch_sampler(
                dataloader_cfg["batch_sampler"], dataset
            )
        return dataloader_cfg

    def _build_dataloader(self, dataset, split, set_batch_size=False):
        dataloader_cfg = copy.deepcopy(self.dataloader_cfg.get(split, {}))
        if set_batch_size:
            dataloader_cfg["batch_size"] = self.batch_size
        dataloader_cfg["collate_fn"] = self._build_collate_fn(
            dataloader_cfg.get("collate_fn", {}), dataset
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

    def setup(self, stage=None):
        super().setup(stage=stage)

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
        for indices, fold_name in zip(
            self.splits[fold_index], [self.TrainSplit, self.ValidateSplit]
        ):
            self.folds[fold_name] = Subset(self.dataset, indices)

        for fold_name in [self.TestSplit, self.PredictSplit]:
            self.folds[fold_name] = self.folds[self.ValidateSplit]

    def train_dataloader(self):
        return self._dataloader(self.TrainSplit)

    def val_dataloader(self):
        return self._dataloader(self.ValidateSplit)

    def test_dataloader(self):
        return self._dataloader(self.TestSplit)

    def predict_dataloader(self):
        return self._dataloader(self.PredictSplit)
