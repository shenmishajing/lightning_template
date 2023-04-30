from typing import Optional

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback


class SaveAndLogConfigCallback(SaveConfigCallback):
    """Saves and logs a LightningCLI config to the log_dir when training starts."""

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None
    ) -> None:
        super().setup(trainer, pl_module, stage)
        if trainer.logger is not None:
            trainer.logger.log_hyperparams(self.config.as_dict())
