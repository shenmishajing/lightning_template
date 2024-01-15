import os
import re
from functools import partial
from typing import Optional, Union

from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.loggers.wandb import WandbLogger


class SetWandbLoggerCallback(Callback):
    """Set wandb logger when training starts."""

    def __init__(
        self,
        log_code_cfg: Optional[Union[dict, bool]] = None,
        watch_model_cfg: Optional[Union[dict, bool]] = None,
    ):
        self.log_code_cfg = log_code_cfg if log_code_cfg is not None else {}
        self.watch_model_cfg = watch_model_cfg if watch_model_cfg is not None else {}

        if isinstance(self.log_code_cfg, dict):
            default_log_code_cfg = {
                "name": "code",
                "root": "code_dirs" if os.path.exists("code_dirs") else ".",
                "include_fn": [
                    re.compile(p)
                    for p in ["\.py$", "\.yaml$", "\.yml$", "\.toml$", "\.sh$", "\.md$"]
                ],
                "exclude_fn": [
                    re.compile(p)
                    for p in ["data/", "work_dirs/", "wandb/", "__pycache__/"]
                ],
            }
            for key, value in default_log_code_cfg.items():
                self.log_code_cfg.setdefault(key, value)
            for name in ["include_fn", "exclude_fn"]:
                if name in self.log_code_cfg:

                    def _is_match(path, root=None, patterns=None):
                        return any(p.search(path) is not None for p in patterns)

                    self.log_code_cfg[name] = partial(
                        _is_match, patterns=self.log_code_cfg[name]
                    )

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None
    ) -> None:
        if trainer.logger is not None and isinstance(trainer.logger, WandbLogger):
            if isinstance(self.watch_model_cfg, dict):
                trainer.logger.watch(model=pl_module, **self.watch_model_cfg)
            if isinstance(self.log_code_cfg, dict):
                trainer.logger.experiment.log_code(**self.log_code_cfg)
