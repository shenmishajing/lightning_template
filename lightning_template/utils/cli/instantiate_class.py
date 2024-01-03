from typing import Dict, List

from jsonargparse import Namespace
from lightning.pytorch.cli import instantiate_class


def is_subclass_spec(val):
    is_class = isinstance(val, (dict, Namespace)) and "class_path" in val
    if is_class:
        keys = getattr(val, "__dict__", val).keys()
        is_class = (
            len(set(keys) - {"class_path", "init_args", "dict_kwargs", "__path__"}) == 0
        )
    return is_class


def recursive_instantate_class(config):
    if isinstance(config, Dict):
        for k, v in config.items():
            config[k] = recursive_instantate_class(v)
        if is_subclass_spec(config):
            return instantiate_class(tuple(), config)
        else:
            return config
    elif isinstance(config, List):
        return [recursive_instantate_class(v) for v in config]
    else:
        return config
