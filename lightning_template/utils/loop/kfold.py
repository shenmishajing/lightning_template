import os
import os.path as osp
from copy import deepcopy
from typing import Any, Dict

from lightning.pytorch.loops import _FitLoop, _Loop
from lightning.pytorch.trainer.states import TrainerFn


class KFoldLoop(_Loop):
    def __init__(self, num_folds: int, fit_loop: _FitLoop = None) -> None:
        super().__init__()
        self.num_folds = num_folds
        self.current_fold = 0
        self._export_path = None
        self.fit_loop = fit_loop
        self.lightning_module_state_dict = None

    @property
    def export_path(self) -> str:
        return os.path.join(self.trainer.log_dir, "fold_checkpoints")

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        self.trainer.datamodule.setup_folds(self.num_folds)
        self.lightning_module_state_dict = deepcopy(
            self.trainer.lightning_module.state_dict()
        )

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        self.trainer.logger.log_metrics(
            {"current_fold": self.current_fold}, step=self.trainer.global_step
        )
        self.trainer.datamodule.setup_fold_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()
        self.current_fold += 1  # increment fold tracking number.

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        self.trainer.save_checkpoint(
            osp.join(self.export_path, f"model_fold_{self.current_fold}.pt")
        )
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.accelerator.setup_optimizers(self.trainer)

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True
        self.fit_loop.epoch_progress.reset_on_run()

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]
