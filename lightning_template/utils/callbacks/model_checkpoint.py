import os
import shutil
from typing import Dict, Optional

import lightning.pytorch as pl
import torch
from fsspec.utils import get_protocol
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.types import _METRIC


class ModelCheckpointWithLinkBest(ModelCheckpoint):
    CHECKPOINT_NAME_BEST = "best"

    def __init__(self, save_best: Optional[bool] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_best = save_best

    def setup(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
    ) -> None:
        if trainer.log_dir:
            self.dirpath = os.path.join(trainer.log_dir, "checkpoints")
        super().setup(trainer, pl_module, stage)

    def _update_best_and_save(
        self,
        current: torch.Tensor,
        trainer: "pl.Trainer",
        monitor_candidates: Dict[str, _METRIC],
    ) -> None:
        old_best_model_path = self.best_model_path
        super()._update_best_and_save(current, trainer, monitor_candidates)
        if old_best_model_path != self.best_model_path:
            self._save_best_checkpoint(trainer, monitor_candidates)

    @staticmethod
    def _link_checkpoint(trainer: "pl.Trainer", filepath: str, linkpath: str) -> None:
        if trainer.is_global_zero:
            if os.path.islink(linkpath) or os.path.isfile(linkpath):
                os.remove(linkpath)
            elif os.path.isdir(linkpath):
                shutil.rmtree(linkpath)
            try:
                os.symlink(
                    os.path.relpath(filepath, os.path.dirname(linkpath)), linkpath
                )
            except OSError:
                # on Windows, special permissions are required to create symbolic links as a regular user
                # fall back to copying the file
                shutil.copy(filepath, linkpath)
        trainer.strategy.barrier()

    def _save_best_checkpoint(
        self, trainer: "pl.Trainer", monitor_candidates: Dict[str, _METRIC]
    ) -> None:
        if not self.save_best:
            return

        filepath = self.format_checkpoint_name(
            monitor_candidates, self.CHECKPOINT_NAME_BEST
        )

        if get_protocol(str(filepath)) == "file" and self.best_model_path:
            self._link_checkpoint(trainer, self.best_model_path, filepath)
        else:
            self._save_checkpoint(trainer, filepath)
