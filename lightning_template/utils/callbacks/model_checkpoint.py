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

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        if trainer.is_global_zero:
            if os.path.islink(filepath) or os.path.isfile(filepath):
                os.remove(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)
        super()._save_checkpoint(trainer, filepath)

    def _save_last_checkpoint(
        self, trainer: "pl.Trainer", monitor_candidates: Dict[str, torch.Tensor]
    ) -> None:
        super()._save_last_checkpoint(trainer, monitor_candidates)
        self._last_checkpoint_saved = None

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
