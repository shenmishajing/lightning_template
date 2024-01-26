import copy
import string
from collections import OrderedDict
from typing import List, Mapping

from ..cli.argument_parsers import deep_update


class SplitNameMixin:
    TrainSplit = "fit"
    ValidateSplit = "val"
    TestSplit = "test"
    PredictSplit = "predict"

    SplitNameMap = OrderedDict()
    SplitNameMap[TrainSplit] = "train"
    SplitNameMap[ValidateSplit] = "val"
    SplitNameMap[TestSplit] = "val"
    SplitNameMap[PredictSplit] = "val"

    def __init__(self) -> None:
        super().__init__()
        self.split_names = list(self.SplitNameMap.keys())

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

    @staticmethod
    def substitute_split_name(
        cur_cfg,
        split_prefix,
        split_attr,
        split_attr_split_str,
        split_name_map,
        split_name,
        *args,
        **kwargs,
    ):
        if split_prefix is not None:
            for s in split_prefix.split(split_attr_split_str):
                if s not in cur_cfg:
                    cur_cfg[s] = {}
                cur_cfg = cur_cfg[s]

        split_attr = split_attr.split(split_attr_split_str)
        for s in split_attr[:-1]:
            cur_cfg = cur_cfg[s]
        split_attr = split_attr[-1]
        cur_cfg[split_attr] = string.Template(
            cur_cfg.get(split_attr, "$split")
        ).safe_substitute(split=split_name_map[split_name])

    def get_split_config(self, config):
        if isinstance(config, Mapping):
            if all([config.get(name) is None for name in self.split_names]):
                res = {self.split_names[0]: config}
                for name in self.split_names[1:]:
                    res[name] = copy.deepcopy(config)
                return res
            else:
                res = {}
                last_config = None
                for name in self.split_names:
                    if last_config is None:
                        res[name] = config[name]
                    else:
                        if isinstance(config.get(name, {}), List):
                            res[name] = []
                            for i in range(len(config[name])):
                                res[name].append(
                                    deep_update(
                                        copy.deepcopy(last_config), config[name][i]
                                    )
                                )
                                last_config = res[name][i]
                        else:
                            res[name] = deep_update(
                                copy.deepcopy(last_config), config.get(name, {})
                            )

                    if isinstance(res[name], List):
                        last_config = res[name][0]
                    else:
                        last_config = res[name]

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
                            if isinstance(res[name], list):
                                for cur_cfg in res[name]:
                                    self.substitute_split_name(
                                        cur_cfg,
                                        split_attr=split_attr,
                                        split_name=name,
                                        **config,
                                    )
                            else:
                                self.substitute_split_name(
                                    res[name],
                                    split_attr=split_attr,
                                    split_name=name,
                                    **config,
                                )
                return res
        else:
            res = {self.split_names[0]: config if config else {}}
            for name in self.split_names[1:]:
                res[name] = copy.deepcopy(config) if config else {}
            return res
