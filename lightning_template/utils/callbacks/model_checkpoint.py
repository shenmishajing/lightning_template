import os
from typing import Dict, Optional

import lightning.pytorch as pl
import torch
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
        super()._update_best_and_save(current, trainer, monitor_candidates)
        self._save_best_checkpoint(trainer, monitor_candidates)

    def _save_best_checkpoint(
        self, trainer: "pl.Trainer", monitor_candidates: Dict[str, _METRIC]
    ) -> None:
        if not self.save_best:
            return

        filepath = self.format_checkpoint_name(
            monitor_candidates, self.CHECKPOINT_NAME_BEST
        )

        if trainer.is_global_zero:
            if self._fs.lexists(filepath):
                self._fs.rm_file(filepath)
            if self._fs.protocol == "file":
                os.symlink(os.path.basename(self.best_model_path), filepath)
            else:
                self._fs.cp_file(self.best_model_path, filepath)
