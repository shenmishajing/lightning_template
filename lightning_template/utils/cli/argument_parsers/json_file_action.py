import json
from argparse import Action
from typing import List, Optional

from jsonargparse import LoggerProperty, Path, get_config_read_mode


class ActionJsonFile(LoggerProperty, Action):
    """Action to indicate that an argument is a configuration file or a configuration
    string in json format."""

    def __call__(self, parser, cfg, values, option_string=None):
        try:
            values = Optional[Path] = Path(
                values, mode=get_config_read_mode()
            ).get_content()
        except TypeError:
            pass
        cfg_file = json.loads(values)
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
