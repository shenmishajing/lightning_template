import copy
import string
from typing import List, Mapping

from ..cli.argument_parsers import deep_update


class SplitNameMixin:
    TrainSplit = "fit"
    ValidateSplit = "val"
    TestSplit = "test"
    PredictSplit = "predict"

    SplitNameMap = {
        TrainSplit: "train",
        ValidateSplit: "val",
        TestSplit: "val",
        PredictSplit: "val",
    }

    def __init__(self) -> None:
        super().__init__()
        self.split_names = [
            self.TrainSplit,
            self.ValidateSplit,
            self.TestSplit,
            self.PredictSplit,
        ]

    def _get_split_names(self, stage=None):
        if self.trainer.overfit_batches > 0:
            split_names = [self.TrainSplit]
        elif stage is None:
            split_names = self.split_names
        elif stage == "fit":
            split_names = [self.TrainSplit, self.ValidateSplit]
        elif stage == "validate":
            split_names = [self.ValidateSplit]
        else:
            split_names = [stage.lower()]
        return split_names

    def setup(self, stage=None):
        super().setup(stage=stage)
        self.split_names = self._get_split_names(stage)

    def get_split_config(self, config):
        if isinstance(config, Mapping):
            if all([config.get(name) is None for name in self.split_names]):
                return {name: copy.deepcopy(config) for name in self.split_names}
            else:
                res = {}
                last_name = None
                for name in self.split_names:
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

                    if "split_name_map" not in config:
                        config["split_name_map"] = {}
                    for k, v in self.SplitNameMap.items():
                        config["split_name_map"].setdefault(k, v)

                    config.setdefault("split_prefix", "init_args")
                    config.setdefault("split_attr_split_str", ".")

                    for name in self.split_names:
                        for split_attr in config["split_format_to"]:
                            cur_cfg = res[name]
                            if config["split_prefix"] is not None:
                                for s in config["split_prefix"].split(
                                    config["split_attr_split_str"]
                                ):
                                    if s not in cur_cfg:
                                        cur_cfg[s] = {}
                                    cur_cfg = cur_cfg[s]

                            split_attr = split_attr.split(
                                config["split_attr_split_str"]
                            )
                            for s in split_attr[:-1]:
                                cur_cfg = cur_cfg[s]
                            split_attr = split_attr[-1]
                            cur_cfg[split_attr] = string.Template(
                                cur_cfg.get(split_attr, "$split")
                            ).safe_substitute(split=config["split_name_map"][name])
                return res
        else:
            return {
                name: copy.deepcopy(config) if config else {}
                for name in self.split_names
            }
