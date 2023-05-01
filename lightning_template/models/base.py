import os
import shutil
from abc import ABC
from typing import Mapping

import torch
from lightning.pytorch import LightningModule as _LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only


def _flatten_dict_gen(log_dict, prefix="train", sep="/"):
    for k, v in log_dict.items():
        new_key = prefix + sep + k if prefix else k
        if isinstance(v, Mapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(log_dict, prefix="train", sep="/"):
    return dict(_flatten_dict_gen(log_dict, prefix, sep))


class LightningModule(_LightningModule, ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_weights=None,
        predict_tasks=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.model = model
        self.loss_weights = loss_weights

        if predict_tasks is None:
            predict_tasks = []
        elif isinstance(predict_tasks, str):
            predict_tasks = [predict_tasks]

        for task in predict_tasks:
            assert hasattr(self, "predict_" + task), f"task {task} is not supported!"

        self.predict_tasks = {task: None for task in predict_tasks}

        # leave for auto lr finder
        self.lr = None
        self.automatic_lr_schedule = True
        self.manual_step_scedulers = []

    def optimizer_step(self, *args, **kwargs) -> None:
        # update params
        super().optimizer_step(*args, **kwargs)

        # manual step lr scheduler
        for scheduler in self.manual_step_scedulers:
            if self.trainer.global_step % scheduler["frequency"] == 0:
                scheduler["scheduler"].step()

    @staticmethod
    def flatten_dict(log_dict, prefix="train", sep="/"):
        return flatten_dict(log_dict, prefix, sep)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _loss_step(self, batch, output, *args, **kwargs):
        return output

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

    def forward_step(self, batch, *args, split="val", **kwargs):
        loss_dict = self.loss_step(batch, self(batch))
        self.log_dict(self.flatten_dict(loss_dict, split), sync_dist=True)
        return loss_dict

    def on_forward_epoch_end(self, *args, **kwargs):
        pass

    def training_step(self, batch, *args, **kwargs):
        loss_dict = self.loss_step(batch, self(batch))
        self.log_dict(self.flatten_dict(loss_dict))
        return loss_dict

    def validation_step(self, *args, **kwargs):
        return self.forward_step(split="val", *args, **kwargs)

    def on_validation_epoch_end(self, *args, **kwargs):
        return self.on_forward_epoch_end(split="val", *args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.forward_step(split="test", *args, **kwargs)

    def on_test_epoch_end(self, *args, **kwargs):
        return self.on_forward_epoch_end(split="test", *args, **kwargs)

    @staticmethod
    @rank_zero_only
    def rm_and_create(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    def on_predict_epoch_start(self) -> None:
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(self.trainer.ckpt_path)), "visualization"
        )

        for name in self.predict_tasks:
            path = os.path.join(output_path, name)
            self.rm_and_create(path)
            self.predict_tasks[name] = path

    def predict_forward(self, *args, **kwargs):
        return {}

    def predict_step(self, *args, **kwargs):
        res = self.predict_forward(*args, **kwargs)
        for name, path in self.predict_tasks.items():
            getattr(self, "predict_" + name)(output_path=path, *args, **kwargs, **res)
