import json
from typing import List, Optional

from jsonargparse import Path, get_config_read_mode
from jsonargparse.actions import Action, FilesCompleterMethod, _ActionSubCommands


class ActionJsonFile(Action, FilesCompleterMethod):
    """Action to indicate that an argument is a configuration file or a configuration string in json format."""

    def __init__(self, **kwargs):
        """Initializer for ActionJsonFile instance."""
        if "default" in kwargs:
            raise ValueError(
                "default not allowed for ActionJsonFile, use default_config_files."
            )
        opt_name = kwargs["option_strings"]
        opt_name = (
            opt_name[0]
            if len(opt_name) == 1
            else [x for x in opt_name if x[0:2] == "--"][0]
        )
        if "." in opt_name:
            raise ValueError("ActionJsonFile must be a top level option.")
        if "help" not in kwargs:
            kwargs[
                "help"
            ] = "Path to a configuration file in json format or a configuration string in json format."
        super().__init__(**kwargs)

    def __call__(self, parser, cfg, values, option_string=None):
        """Parses the given configuration and adds all the corresponding keys to the namespace.

        Raises:
            TypeError: If there are problems parsing the configuration.
        """
        self.apply_config(parser, cfg, self.dest, values)

    @staticmethod
    def apply_config(parser, cfg, dest, value) -> None:
        with _ActionSubCommands.not_single_subcommand():
            try:
                cfg_path: Optional[Path] = Path(value, mode=get_config_read_mode())
                value = cfg_path.get_content()
            except TypeError:
                pass
            cfg_file = json.loads(value)
            for key, value in cfg_file.items():
                *prefix_keys, last_key = key.split("/")
                cur_cfg = cfg
                for prefix_key in prefix_keys:
                    if prefix_key:
                        if isinstance(cur_cfg, List):
                            prefix_key = int(prefix_key)
                        cur_cfg = cur_cfg[prefix_key]
                if isinstance(cur_cfg, List):
                    last_key = int(last_key)
                cur_cfg[last_key] = value
