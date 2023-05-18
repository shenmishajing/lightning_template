from typing import Any, Optional, Sequence

from lightning.pytorch.loggers.wandb import WandbLogger


class WandbNamedLogger(WandbLogger):
    def __init__(
        self,
        entity: Optional[str] = None,
        tags: Optional[Sequence] = None,
        group: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, entity=entity, tags=tags, group=group, **kwargs)

    @property
    def name(self) -> Optional[str]:
        """Gets the name of the experiment.

        Returns:
            The name of the experiment if the experiment exists else the name given to the constructor.
        """
        # don't create an experiment if we don't have one
        return self._experiment.name if self._experiment else self._name
