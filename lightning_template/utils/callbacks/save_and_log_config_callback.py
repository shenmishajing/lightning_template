import os

from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.trainer.states import TrainerFn


class SaveAndLogConfigCallback(SaveConfigCallback):
    """Saves and logs a LightningCLI config to the log_dir when training starts."""

    def __init__(
        self,
        *args,
        save_to_log_dir: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, save_to_log_dir=save_to_log_dir, **kwargs)

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
        # save local config file
        if stage == TrainerFn.FITTING:
            trainer.checkpoint_callback.setup(trainer, pl_module, stage)
            log_dir = os.path.dirname(trainer.checkpoint_callback.dirpath)
            config_path = os.path.join(log_dir, self.config_filename)

            fs = get_filesystem(log_dir)
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config,
                config_path,
                skip_none=False,
                overwrite=self.overwrite,
                multifile=self.multifile,
            )

        # log config file
        if trainer.logger is not None:
            trainer.logger.log_hyperparams(self.config.as_dict())
