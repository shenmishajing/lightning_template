import warnings

from torch.optim.lr_scheduler import _LRScheduler


class KeepAndLinearlyDecayLrScheduler(_LRScheduler):
    """Keep lr first then decay learning rate linearly in optimizer.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        keep_epochs: target keep epoch.
        decay_epochs: target decay epoch.
    """

    def __init__(
        self, optimizer, keep_epochs, decay_epochs, last_epoch=-1, verbose=False
    ):
        self.keep_epochs = keep_epochs
        self.decay_epochs = decay_epochs
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch < self.keep_epochs:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            base_lr
            * max(0, 1 - (self.last_epoch + 1 - self.keep_epochs) / self.decay_epochs)
            for base_lr in self.base_lrs
        ]
