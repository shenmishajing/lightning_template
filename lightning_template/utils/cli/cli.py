import os
import time
from types import MethodType
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from lightning.pytorch.cli import (
    LightningArgumentParser,
    LightningCLI as _LightningCLI,
    SaveConfigCallback,
)

from lightning_template.utils.callbacks.save_and_log_config_callback import SaveAndLogConfigCallback
from lightning_template.utils.optim import get_configure_optimizers_method

from .argument_parsers import ActionJsonFile
from .trainer import Trainer, _Trainer


class LightningCLI(_LightningCLI):
    def __init__(
        self,
        save_config_callback: Optional[
            Type[SaveConfigCallback]
        ] = SaveAndLogConfigCallback,
        trainer_class: Union[Type[_Trainer], Callable[..., _Trainer]] = Trainer,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            save_config_callback=save_config_callback,
            trainer_class=trainer_class,
            *args,
            **kwargs
        )

    def _setup_parser_kwargs(
        self, *args, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        parser_kwargs = {"parser_mode": "yaml_with_merge"}
        main_kwargs, subparser_kwargs = super()._setup_parser_kwargs(*args, **kwargs)

        for k, v in parser_kwargs.items():
            main_kwargs.setdefault(k, v)
        for subcommand in self.subcommands():
            for k, v in parser_kwargs.items():
                if subcommand not in subparser_kwargs:
                    subparser_kwargs[subcommand] = {}
                subparser_kwargs[subcommand].setdefault(k, v)

        return main_kwargs, subparser_kwargs

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_default_arguments_to_parser(parser)
        parser.add_argument("--json-file", action=ActionJsonFile)
        parser.add_argument(
            "--optimizer_config",
            type=Optional[Dict],
            default=None,
            help="Configuration for the optimizers and lr schedulers.",
        )

    def before_instantiate_classes(self) -> None:
        """Implement to run some code before instantiating the classes."""
        super().before_instantiate_classes()
        config = (
            self.config
            if "subcommand" not in self.config
            else self.config[self.config["subcommand"]]
        )
        name = os.path.splitext(os.path.split(config["config"][0].abs_path)[1])[0]
        version = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
        if (
            config.get("trainer") is not None
            and config["trainer"].get("logger") is not None
        ):
            if config["trainer"]["logger"].get("init_args") is None:
                config["trainer"]["logger"]["init_args"] = {}
            save_dir = config["trainer"]["logger"]["init_args"].get("save_dir")
            save_dir = os.path.join(
                save_dir if save_dir else "work_dirs", name, version
            )
            os.makedirs(save_dir, exist_ok=True)
            config["trainer"]["logger"]["init_args"]["save_dir"] = save_dir
            for k, v in {"name": name, "version": version}.items():
                if config["trainer"]["logger"]["init_args"].get(k) is None:
                    config["trainer"]["logger"]["init_args"][k] = v

    def _add_configure_optimizers_method_to_model(self, *args, **kwargs) -> None:
        super()._add_configure_optimizers_method_to_model(*args, **kwargs)
        optimizer_config = self._get(self.config_init, "optimizer_config")
        if optimizer_config:
            self.model.configure_optimizers = MethodType(
                get_configure_optimizers_method(optimizer_config), self.model
            )
