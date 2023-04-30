from typing import Optional

from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_warn
from torch.multiprocessing import get_all_sharing_strategies, set_sharing_strategy


class SetSharingStrategyCallback(Callback):
    """Set sharing strategy when training starts."""

    def __init__(
        self,
        strategy: Optional[str] = "file_descriptor",
    ):
        self.strategy = strategy

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None
    ) -> None:
        all_strategies = get_all_sharing_strategies()
        if self.strategy in all_strategies:
            set_sharing_strategy(self.strategy)
        else:
            rank_zero_warn(
                f"Sharing strategy {self.strategy} is not supported."
                f"All supported strategy are: {all_strategies}"
            )
