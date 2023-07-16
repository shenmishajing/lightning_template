import datetime
import time
from typing import Any, Dict, Optional, Union

import lightning.pytorch as pl
from lightning.pytorch.callbacks.progress.rich_progress import (
    RichProgressBar as _RichProgressBar,
)


class RichProgressBar(_RichProgressBar):
    def __init__(
        self,
        show_version: Optional[bool] = False,
        show_eta_time: Optional[bool] = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.show_version = show_version
        self.show_eta_time = show_eta_time

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        if self.show_eta_time:
            self.start_time = time.time()
            self.start_epoch = trainer.current_epoch

    def get_metrics(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> Dict[str, Union[int, str]]:
        items = super().get_metrics(trainer, pl_module)

        if not self.show_version:
            # don't show the version number
            items.pop("v_num", None)

        if (
            self.show_eta_time
            and self.trainer.training
            and self.trainer.max_epochs is not None
            and self.progress
        ):
            during_time = time.time() - self.start_time
            time_sec_avg = during_time / (
                (self.trainer.current_epoch - self.start_epoch)
                * self.total_train_batches
                + self.train_progress_bar.completed
            )
            eta_time = time_sec_avg * (
                (self.trainer.max_epochs - self.trainer.current_epoch)
                * self.total_train_batches
                - self.train_progress_bar.completed
            )
            items["ETA"] = str(datetime.timedelta(seconds=int(eta_time)))
        return items
