from functools import partial
from typing import Optional

from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.loggers.wandb import WandbLogger


class SetWandbLoggerCallback(Callback):
    """Set wandb logger when training starts."""

    def __init__(
        self,
        log_code_cfg: Optional[dict] = None,
        watch_model_cfg: Optional[dict] = None,
    ):
        self.log_code_cfg = log_code_cfg if log_code_cfg is not None else {}
        self.watch_model_cfg = watch_model_cfg if watch_model_cfg is not None else {}

        default_log_code_cfg = {
            "name": "code",
            "include_fn": [".py", ".yaml", ".yml", ".sh", ".md"],
        }
        for key, value in default_log_code_cfg.items():
            self.log_code_cfg.setdefault(key, value)
        for name in ["include_fn", "exclude_fn"]:
            if name in self.log_code_cfg:
                self.log_code_cfg[name] = partial(
                    lambda path, exts: any([path.endswith(ext) for ext in exts]),
                    exts=self.log_code_cfg[name],
                )

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None
    ) -> None:
        if trainer.logger is not None and isinstance(trainer.logger, WandbLogger):
            trainer.logger.watch(model=pl_module, **self.watch_model_cfg)
            trainer.logger.experiment.log_code(**self.log_code_cfg)
