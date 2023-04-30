import warnings

from torch.optim.lr_scheduler import _LRScheduler


class WarmupScheduler(_LRScheduler):
    """Warm-up(increasing) learning rate in optimizer.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_iters: target warm up epoch.
    """

    def __init__(
        self,
        optimizer,
        warmup_iters,
        warmup_ratio=0.1,
        warmup_mode="linear",
        last_epoch=-1,
        verbose=False,
    ):
        # validate the "warmup" argument
        assert warmup_mode is not None and warmup_mode in [
            "constant",
            "linear",
            "exp",
        ], f'"{warmup_mode}" is not a supported type for warming up, valid types are "constant" and "linear"'
        assert warmup_iters > 0, '"warmup_iters" must be a positive integer'
        assert 0 < warmup_ratio <= 1.0, '"warmup_ratio" must be in range (0,1]'

        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_mode = warmup_mode
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch <= self.warmup_iters:
            if self.warmup_mode == "constant":
                return [lr * self.warmup_ratio for lr in self.base_lrs]
            elif self.warmup_mode == "linear":
                k = (1 - self.last_epoch / self.warmup_iters) * (1 - self.warmup_ratio)
                return [lr * (1 - k) for lr in self.base_lrs]
            elif self.warmup_mode == "exp":
                k = self.warmup_ratio ** (1 - self.last_epoch / self.warmup_iters)
                return [lr * k for lr in self.base_lrs]
        return [group["lr"] for group in self.optimizer.param_groups]
