from typing import Dict, List

from jsonargparse.typehints import is_subclass_spec
from lightning.pytorch.cli import instantiate_class


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
