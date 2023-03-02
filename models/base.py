import os
import shutil
from abc import ABC

import torch
from lightning.pytorch import LightningModule as _LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from mmengine.model import BaseModule


class LightningModule(_LightningModule, BaseModule, ABC):
    def __init__(
        self,
        loss_weights=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.automatic_lr_schedule = True
        self.manual_step_scedulers = []
        self._output_paths = []
        self.lr = None
        self.batch_size = None

        self.loss_weights = loss_weights

    def optimizer_step(self, *args, **kwargs) -> None:
        # update params
        super().optimizer_step(*args, **kwargs)

        # manual step lr scheduler
        for scheduler in self.manual_step_scedulers:
            if self.trainer.global_step % scheduler["frequency"] == 0:
                scheduler["scheduler"].step()

    def log(self, *args, batch_size=None, **kwargs):
        if (
            batch_size is None
            and hasattr(self, "batch_size")
            and self.batch_size is not None
        ):
            batch_size = self.batch_size
        super().log(*args, batch_size=batch_size, **kwargs)

    def _dump_init_info(self, *args, **kwargs):
        pass

    @staticmethod
    def add_prefix(log_dict, prefix="train", sep="/"):
        return {f"{prefix}{sep}{k}": v for k, v in log_dict.items()}

    def on_fit_start(self):
        self.init_weights()

    def _loss_step(self, *args, **kwargs):
        raise NotImplementedError

    def loss_step(self, *args, use_loss_weight=True, **kwargs):
        loss = self._loss_step(*args, **kwargs)
        # multi loss weights
        if use_loss_weight and self.loss_weights:
            loss = {
                k: v * (1 if k not in self.loss_weights else self.loss_weights[k])
                for k, v in loss.items()
            }
        # calculate loss
        if "loss" not in loss:
            loss["loss"] = torch.sum(torch.stack(list(loss.values())))
        return loss

    def training_step(self, batch, *args, **kwargs):
        loss_dict = self.loss_step(batch, self(batch))
        self.log_dict(self.add_prefix(loss_dict))
        return loss_dict

    def validation_step(self, batch, *args, **kwargs):
        loss_dict = self.loss_step(batch, self(batch))
        self.log_dict(self.add_prefix(loss_dict, prefix="val"), sync_dist=True)
        return loss_dict

    def test_step(self, batch, *args, **kwargs):
        loss_dict = self.loss_step(batch, self(batch))
        self.log_dict(self.add_prefix(loss_dict, prefix="test"), sync_dist=True)
        return loss_dict

    @staticmethod
    @rank_zero_only
    def rm_and_create(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    @property
    def output_paths(self):
        return self._output_paths

    def on_predict_start(self) -> None:
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(self.trainer.ckpt_path)), "visualization"
        )

        for name in self.output_paths:
            path = os.path.join(output_path, name)
            self.rm_and_create(path)
            setattr(self, name + "_output_path", path)

    def predict_step(self, *args, **kwargs):
        for name in self.output_paths:
            getattr(self, name + "_visualization")(*args, **kwargs)