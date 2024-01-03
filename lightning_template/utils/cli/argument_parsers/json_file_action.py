import json
from typing import List, Optional

from jsonargparse import ActionConfigFile, Path, get_config_read_mode


class ActionJsonFile(ActionConfigFile):
    """Action to indicate that an argument is a configuration file or a configuration
    string in json format."""

    @staticmethod
    def apply_config(parser, cfg, dest, value) -> None:
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
