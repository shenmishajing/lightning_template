import copy
import string
from typing import List, Mapping

from .argument_parsers import deep_update


def get_split_config(config, split_names=None):
    if split_names is None:
        split_names = ["train", "val", "test", "predict"]

    if isinstance(config, Mapping):
        if all([config.get(name) is None for name in split_names]):
            return {name: copy.deepcopy(config) for name in split_names}
        else:
            res = {}
            last_name = None
            for name in split_names:
                if last_name is None:
                    res[name] = copy.deepcopy(config[name])
                else:
                    res[name] = deep_update(
                        copy.deepcopy(res[last_name]), config.get(name, {})
                    )
                last_name = name

            if "split_info" in config and "split_format_to" in config["split_info"]:
                config = config["split_info"]

                if not isinstance(config["split_format_to"], List):
                    config["split_format_to"] = [config["split_format_to"]]

                split_name_map = {
                    "train": "train",
                    "val": "val",
                    "test": "val",
                    "predict": "val",
                }
                split_name_map.update(config.get("split_name_map", {}))
                config["split_name_map"] = split_name_map

                config.setdefault("split_prefix", "init_args")
                config.setdefault("split_attr_split_str", ".")

                for name in split_names:
                    for split_attr in config["split_format_to"]:
                        cur_cfg = res[name]
                        if config["split_prefix"] is not None:
                            for s in config["split_prefix"].split(
                                config["split_attr_split_str"]
                            ):
                                if s not in cur_cfg:
                                    cur_cfg[s] = {}
                                cur_cfg = cur_cfg[s]

                        split_attr = split_attr.split(config["split_attr_split_str"])
                        for s in split_attr[:-1]:
                            cur_cfg = cur_cfg[s]
                        split_attr = split_attr[-1]
                        cur_cfg[split_attr] = string.Template(
                            cur_cfg.get(split_attr, "$split")
                        ).safe_substitute(split=config["split_name_map"][name])
            return res
    else:
        return {name: copy.deepcopy(config) if config else {} for name in split_names}
