import functools
from types import MethodType
from typing import Optional

from lightning.pytorch import Trainer as _Trainer
from lightning.pytorch.utilities.argparse import _defaults_from_env_vars

from ..loop import KFoldLoop


def _use_auto_lr_schedule(old_func):
    @functools.wraps(old_func)
    def wrapper(obj, *args, **kwargs):
        automatic_optimization = obj.trainer.lightning_module.automatic_optimization
        obj.trainer.lightning_module.automatic_optimization |= (
            obj.trainer.lightning_module.automatic_lr_schedule
        )
        old_func(*args, **kwargs)
        obj.trainer.lightning_module.automatic_optimization = automatic_optimization

    return wrapper


class Trainer(_Trainer):
    @_defaults_from_env_vars
    def __init__(self, num_folds: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fit_loop.epoch_loop._update_learning_rates = MethodType(
            _use_auto_lr_schedule(self.fit_loop.epoch_loop._update_learning_rates),
            self.fit_loop.epoch_loop,
        )
        # add kfold cross validation support
        self.num_folds = num_folds
        if self.num_folds is not None and self.num_folds > 1:
            self.fit_loop = KFoldLoop(self.num_folds, self.fit_loop)
