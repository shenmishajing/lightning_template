from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback


class SaveAndLogConfigCallback(SaveConfigCallback):
    """Saves and logs a LightningCLI config to the log_dir when training starts."""

    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """Implement to save the config in some other place additional to the standard
        log_dir.

        Example:
            def save_config(self, trainer, pl_module, stage):
                if isinstance(trainer.logger, Logger):
                    config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
                    trainer.logger.log_hyperparams({"config": config})

        Note:
            This method is only called on rank zero. This allows to implement a custom save config without having to
            worry about ranks or race conditions. Since it only runs on rank zero, any collective call will make the
            process hang waiting for a broadcast. If you need to make collective calls, implement the setup method
            instead.
        """
        if trainer.logger is not None:
            trainer.logger.log_hyperparams(self.config.as_dict())
