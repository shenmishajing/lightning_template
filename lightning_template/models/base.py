import os
import shutil
from typing import Mapping

import torch
from lightning.pytorch import LightningModule as _LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from lightning_template.utils.cli import recursive_instantate_class
from lightning_template.utils.mixin import SplitNameMixin


class LightningModule(SplitNameMixin, _LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        evaluator_cfg: dict = None,
        loss_weights=None,
        predict_tasks=None,
        predict_path=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.model = model
        self.evaluators = torch.nn.ModuleDict()
        self.loss_weights = loss_weights

        self.evaluator_cfg = self.get_split_config(evaluator_cfg)

        if predict_tasks is None:
            predict_tasks = []
        elif isinstance(predict_tasks, str):
            predict_tasks = [predict_tasks]

        for task in predict_tasks:
            assert hasattr(self, "predict_" + task), f"task {task} is not supported!"

        self.predict_tasks = {task: predict_path for task in predict_tasks}

        # leave for auto lr finder
        self.lr = None
        self.automatic_lr_schedule = True
        self.manual_step_scedulers = []

    def _build_evaluator(self, split):
        if split in self.evaluator_cfg and self.evaluator_cfg[split]:
            self.evaluators[split] = recursive_instantate_class(
                self.evaluator_cfg[split]
            )
        else:
            self.evaluators[split] = None

    def setup(self, stage=None):
        super().setup(stage=stage)

        for name in self.split_names:
            self._build_evaluator(name)

    def optimizer_step(self, *args, **kwargs) -> None:
        # update params
        super().optimizer_step(*args, **kwargs)

        # manual step lr scheduler
        for scheduler in self.manual_step_scedulers:
            if self.trainer.global_step % scheduler["frequency"] == 0:
                scheduler["scheduler"].step()

    @staticmethod
    def flatten_dict(log_dict, prefix, sep="/"):
        res_dict = {}

        for k, v in log_dict.items():
            new_key = prefix + sep + k if prefix else k
            if isinstance(v, Mapping):
                res_dict[new_key] = LightningModule.flatten_dict(v, new_key, sep=sep)
            else:
                res_dict[new_key] = v
        return res_dict

    def forward(self, batch, *args, **kwargs):
        return self.model(batch)

    def _loss_step(self, batch, output, *args, **kwargs):
        if not isinstance(output, Mapping):
            return {"loss": output}
        elif "log_dict" in output:
            return output["log_dict"]
        elif hasattr(self.model, "loss_step"):
            return self.model.loss_step(batch, output)
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
            loss["loss"] = torch.sum(
                torch.stack([v for k, v in loss.items() if "loss" in k])
            )
        return loss

    def metric_step(self, batch, output, *args, split, **kwargs):
        if self.evaluators[split]:
            self.evaluators[split].update(output["pred"], output["target"])

    def on_metric_epoch_end(self, *args, split, **kwargs):
        if self.evaluators[split]:
            result = self.evaluators[split].compute()
            self.evaluators[split].reset()
            if not isinstance(result, dict):
                result = {self.evaluators[split].__class__.__name__: result}
            return result

    def forward_step(self, batch, *args, split, **kwargs):
        # forward
        output = self(batch, *args, split=split, **kwargs)

        # loss
        log_dict = self.loss_step(batch, output, *args, split=split, **kwargs)

        # metrics
        metrics = self.metric_step(batch, output, split=split, *args, **kwargs)
        if metrics:
            log_dict.update(metrics)

        # log
        self.log_dict(
            self.flatten_dict(log_dict, split), sync_dist=split != self.TrainSplit
        )

        # return loss
        return log_dict

    def on_forward_epoch_end(self, split, *args, **kwargs):
        log_dict = self.on_metric_epoch_end(split=split, *args, **kwargs)

        if log_dict:
            self.log_dict(
                self.flatten_dict(log_dict, split), sync_dist=split != self.TrainSplit
            )

    def training_step(self, *args, **kwargs):
        return self.forward_step(split=self.TrainSplit, *args, **kwargs)

    def on_train_epoch_end(self, *args, **kwargs):
        return self.on_forward_epoch_end(split=self.TrainSplit, *args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return self.forward_step(split=self.ValidateSplit, *args, **kwargs)

    def on_validation_epoch_end(self, *args, **kwargs):
        return self.on_forward_epoch_end(split=self.ValidateSplit, *args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.forward_step(split=self.TestSplit, *args, **kwargs)

    def on_test_epoch_end(self, *args, **kwargs):
        return self.on_forward_epoch_end(split=self.TestSplit, *args, **kwargs)

    @staticmethod
    @rank_zero_only
    def rm_and_create(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    def on_predict_epoch_start(self) -> None:
        if self.trainer.ckpt_path:
            output_path = os.path.join(
                os.path.dirname(os.path.dirname(self.trainer.ckpt_path)),
                "visualization",
            )
        else:
            output_path = None

        for name in self.predict_tasks:
            if self.predict_tasks[name] is None:
                if output_path is None:
                    raise ValueError(
                        "predict_path is None, please set predict_path or pass ckpt_path"
                    )

                self.predict_tasks[name] = output_path

            self.predict_tasks[name] = os.path.join(self.predict_tasks[name], name)
            self.rm_and_create(self.predict_tasks[name])

    def predict_forward(self, *args, **kwargs):
        return {}

    def predict_step(self, *args, **kwargs):
        res = self.predict_forward(*args, **kwargs)
        for name, path in self.predict_tasks.items():
            getattr(self, "predict_" + name)(output_path=path, *args, **kwargs, **res)
