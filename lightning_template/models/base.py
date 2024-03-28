import os
import shutil
from typing import List, Mapping, Optional, Union

import torch
from lightning.pytorch import LightningModule as _LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from lightning_template.utils.cli import recursive_instantate_class
from lightning_template.utils.mixin import SplitNameMixin
from lightning_template.utils.optim.configure_optimizers import get_parameters


class LightningModule(SplitNameMixin, _LightningModule):
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        ckpt_path: Optional[Union[str, List[str]]] = None,
        finetune_cfg: Optional[Union[str, List[str], Mapping]] = None,
        evaluator_cfg: Mapping = None,
        evaluator_as_submodule: bool = True,
        loss_weights=None,
        predict_tasks: Optional[List[str]] = None,
        predict_path: str = "prediction",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.model = model
        if isinstance(ckpt_path, str):
            ckpt_path = [ckpt_path]
        self.ckpt_path = ckpt_path
        self.evaluators = {}
        self.loss_weights = loss_weights

        self.evaluator_cfg = self.get_split_config(evaluator_cfg)
        self.evaluate_as_submodule = evaluator_as_submodule

        if finetune_cfg is not None:
            if isinstance(finetune_cfg, str):
                finetune_cfg = {finetune_cfg}
            if isinstance(finetune_cfg, list):
                finetune_cfg = set(finetune_cfg)
            if isinstance(finetune_cfg, set):
                finetune_cfg = {"finetune": True, "params": finetune_cfg}
        self.finetune_cfg = finetune_cfg

        if predict_tasks is None:
            predict_tasks = []
        elif isinstance(predict_tasks, str):
            predict_tasks = [predict_tasks]

        for task in predict_tasks:
            assert hasattr(self, "predict_" + task), f"task {task} is not supported!"

        self.predict_tasks = predict_tasks
        self.predict_path = predict_path

        # leave for auto lr finder
        self.lr = None
        self.automatic_lr_schedule = True
        self.manual_step_scedulers = []
        self.model_not_configured = True

    def build_model(self):
        pass

    def configure_model(self):
        super().configure_model()

        if self.model_not_configured:
            self.build_model()

            if self.ckpt_path is not None:
                for p in self.ckpt_path:
                    if os.path.exists(p):
                        checkpoint = torch.load(p, map_location="cpu")
                        self.on_load_checkpoint(checkpoint)
                        self.load_state_dict(checkpoint["state_dict"], strict=False)

            if self.finetune_cfg:
                params = get_parameters(
                    self,
                    self.finetune_cfg["params"],
                    finetune_rest=not self.finetune_cfg["finetune"],
                )
                for param in params.values():
                    for p in param:
                        p.requires_grad = self.finetune_cfg["finetune"]

            self.model_not_configured = False

    @staticmethod
    def recursive_parse_modules(module):
        modules = []
        if isinstance(module, torch.nn.Module):
            modules.append(module)
        elif isinstance(module, list):
            for m in module:
                modules.extend(LightningModule.recursive_parse_modules(m))
        elif isinstance(module, dict):
            for m in module.values():
                modules.extend(LightningModule.recursive_parse_modules(m))
        return modules

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
        if self.evaluate_as_submodule:
            self._evaluators = torch.nn.ModuleList(
                self.recursive_parse_modules(self.evaluators)
            )

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

    def _loss_step(self, *args, output, **kwargs):
        if not isinstance(output, Mapping):
            return {"loss": output}
        elif "loss_dict" in output:
            return output["loss_dict"]
        elif self.model is not None and hasattr(self.model, "loss_step"):
            return self.model.loss_step(*args, output=output, **kwargs)
        return output

    def loss_step(self, *args, use_loss_weight=True, **kwargs):
        loss = self._loss_step(*args, **kwargs)
        # multiply loss weights
        if use_loss_weight and self.loss_weights:
            loss = {
                k: v * (1 if k not in self.loss_weights else self.loss_weights[k])
                for k, v in loss.items()
            }
        # calculate loss if necessary
        if "loss" not in loss:
            loss["loss"] = torch.sum(
                torch.stack([v for k, v in loss.items() if "loss" in k])
            )
        return loss

    def update_evaluator(self, evaluator, *args, metrics, **kwargs):
        evaluator.update(**metrics)

    def _metric_step(self, *args, output, **kwargs):
        if isinstance(output, Mapping) and "metric_dict" in output:
            return output["metric_dict"]
        elif self.model is not None and hasattr(self.model, "metric_step"):
            return self.model.metric_step(*args, output, **kwargs)
        return None

    def metric_step(self, *args, dataloader_idx=None, split, **kwargs):
        if self.evaluators[split]:
            metrics = self._metric_step(
                dataloader_idx=dataloader_idx, split=split, *args, **kwargs
            )
            if metrics is not None:
                if dataloader_idx is not None and isinstance(
                    self.evaluators[split], (list, torch.nn.ModuleList)
                ):
                    self.update_evaluator(
                        self.evaluators[split][dataloader_idx],
                        *args,
                        metrics=metrics,
                        dataloader_idx=dataloader_idx,
                        split=split,
                        **kwargs,
                    )
                else:
                    self.update_evaluator(
                        self.evaluators[split],
                        *args,
                        metrics=metrics,
                        dataloader_idx=dataloader_idx,
                        split=split,
                        **kwargs,
                    )

    def _compute_evaluator(self, evaluator, *args, **kwargs):
        result = evaluator.compute()
        evaluator.reset()
        if not isinstance(result, dict):
            result = {evaluator.__class__.__name__: result}
        return result

    def compute_evaluator(self, evaluator, dataloader_idx=None, *args, **kwargs):
        result = self._compute_evaluator(
            evaluator, dataloader_idx=dataloader_idx, *args, **kwargs
        )
        if dataloader_idx is not None:
            result = {f"{k}_{dataloader_idx}": v for k, v in result.items()}
        return result

    def on_metric_epoch_end(self, *args, split, **kwargs):
        if self.evaluators[split]:
            if isinstance(self.evaluators[split], (list, torch.nn.ModuleList)):
                result = {}
                for dataloader_idx, evaluator in enumerate(self.evaluators[split]):
                    result.update(
                        self.compute_evaluator(
                            evaluator,
                            dataloader_idx=dataloader_idx,
                            split=split,
                            *args,
                            **kwargs,
                        )
                    )
            else:
                result = self.compute_evaluator(
                    self.evaluators[split],
                    split=split,
                    *args,
                    **kwargs,
                )
            return result

    def forward_step(self, *args, split, **kwargs):
        # forward
        output = self(split=split, *args, **kwargs)

        # loss
        log_dict = self.loss_step(output=output, split=split, *args, **kwargs)

        # metrics
        metrics = self.metric_step(output=output, split=split, *args, **kwargs)
        if metrics:
            log_dict.update(metrics)

        # log
        self.log_dict(
            self.flatten_dict(log_dict, split),
            batch_size=output.get("batch_size", None)
            if isinstance(output, Mapping)
            else None,
            sync_dist=split != self.TrainSplit,
        )

        # return loss
        return log_dict

    def on_forward_epoch_end(self, *args, split, **kwargs):
        log_dict = self.on_metric_epoch_end(split=split, *args, **kwargs)

        if log_dict:
            batch_size = log_dict.pop("batch_size", None)
            self.log_dict(
                self.flatten_dict(log_dict, split),
                batch_size=batch_size,
                sync_dist=True,
            )

    def training_step(self, batch, batch_idx, dataloader_idx=None, *args, **kwargs):
        return self.forward_step(
            batch,
            batch_idx,
            dataloader_idx=dataloader_idx,
            split=self.TrainSplit,
            *args,
            **kwargs,
        )

    def on_train_epoch_end(self, *args, **kwargs):
        return self.on_forward_epoch_end(split=self.TrainSplit, *args, **kwargs)

    def validation_step(self, batch, batch_idx, dataloader_idx=None, *args, **kwargs):
        return self.forward_step(
            batch,
            batch_idx,
            dataloader_idx=dataloader_idx,
            split=self.ValidateSplit,
            *args,
            **kwargs,
        )

    def on_validation_epoch_end(self, *args, **kwargs):
        return self.on_forward_epoch_end(split=self.ValidateSplit, *args, **kwargs)

    def test_step(self, batch, batch_idx, dataloader_idx=None, *args, **kwargs):
        return self.forward_step(
            batch,
            batch_idx,
            dataloader_idx=dataloader_idx,
            split=self.TestSplit,
            *args,
            **kwargs,
        )

    def on_test_epoch_end(self, *args, **kwargs):
        return self.on_forward_epoch_end(split=self.TestSplit, *args, **kwargs)

    @staticmethod
    @rank_zero_only
    def rm_and_create(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    def on_predict_start(self) -> None:
        if self.trainer.ckpt_path:
            self.predict_path = os.path.join(
                os.path.dirname(os.path.dirname(self.trainer.ckpt_path)),
                self.predict_path,
            )

        for name in self.predict_tasks:
            self.rm_and_create(os.path.join(self.predict_path, name))

    def predict_forward(self, *args, **kwargs):
        return {}

    def predict_step(self, *args, **kwargs):
        res = self.predict_forward(*args, **kwargs)
        for name in self.predict_tasks:
            getattr(self, "predict_" + name)(
                output_path=os.path.join(self.predict_path, name),
                *args,
                **kwargs,
                **res,
            )
