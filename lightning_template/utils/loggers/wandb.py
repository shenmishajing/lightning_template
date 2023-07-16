from typing import Optional

from lightning.pytorch.loggers.wandb import WandbLogger


class WandbNamedLogger(WandbLogger):
    @property
    def name(self) -> Optional[str]:
        """Gets the name of the experiment.

        Returns:
            The name of the experiment if the experiment exists else the name given to the constructor.
        """
        # don't create an experiment if we don't have one
        return self._experiment.name if self._experiment else self._name
