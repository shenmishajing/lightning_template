import copy
import os

from jsonargparse import Path, get_config_read_mode, set_dumper, set_loader
from jsonargparse.loaders_dumpers import DefaultLoader, dumpers, yaml_load
from jsonargparse.util import change_to_path_dir
from yaml.constructor import FullConstructor

from .deep_update import deep_update

DefaultLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", FullConstructor.construct_python_tuple
)


def get_cfg_from_path(cfg_path):
    fpath = Path(cfg_path, mode=get_config_read_mode())
    with change_to_path_dir(fpath):
        cfg_str = fpath.get_content()
        parsed_cfg = yaml_load(cfg_str)
    return parsed_cfg


def parse_config(cfg_file, cfg_path=None, **kwargs):
    if "__import__" in cfg_file:
        cfg_file.pop("__import__")

    for k, v in cfg_file.items():
        if isinstance(v, dict):
            cfg_file[k] = parse_config(v, cfg_path, **kwargs)

    if "__base__" in cfg_file:
        sub_cfg_paths = cfg_file.pop("__base__")
        if sub_cfg_paths is not None:
            if not isinstance(sub_cfg_paths, list):
                sub_cfg_paths = [sub_cfg_paths]
            sub_cfg_paths = [
                sub_cfg_path if isinstance(sub_cfg_path, list) else [sub_cfg_path, ""]
                for sub_cfg_path in sub_cfg_paths
            ]
            if cfg_path is not None:
                sub_cfg_paths = [
                    [
                        os.path.normpath(
                            os.path.join(os.path.dirname(cfg_path), sub_cfg_path[0])
                        )
                        if not os.path.isabs(sub_cfg_path[0])
                        else sub_cfg_path[0],
                        sub_cfg_path[1],
                    ]
                    for sub_cfg_path in sub_cfg_paths
                ]
            sub_cfg_file = {}
            for sub_cfg_path in sub_cfg_paths:
                cur_cfg_file = parse_path(sub_cfg_path[0], **kwargs)
                for key in sub_cfg_path[1].split("."):
                    if key:
                        cur_cfg_file = cur_cfg_file[key]
                sub_cfg_file = deep_update(sub_cfg_file, cur_cfg_file)
            cfg_file = deep_update(sub_cfg_file, cfg_file)

    return cfg_file


def parse_path(cfg_path, seen_cfg=None, **kwargs):
    abs_cfg_path = os.path.abspath(cfg_path)
    if seen_cfg is None:
        seen_cfg = {}
    elif abs_cfg_path in seen_cfg:
        if seen_cfg[abs_cfg_path] is None:
            raise RuntimeError("Circular reference detected in config file")
        else:
            return copy.deepcopy(seen_cfg[abs_cfg_path])

    cfg_file = get_cfg_from_path(cfg_path)
    seen_cfg[abs_cfg_path] = None
    cfg_file = parse_config(cfg_file, cfg_path=cfg_path, seen_cfg=seen_cfg, **kwargs)
    seen_cfg[abs_cfg_path] = copy.deepcopy(cfg_file)
    return cfg_file


def parse_str(cfg_str, cfg_path=None, seen_cfg=None, **kwargs):
    if seen_cfg is None:
        seen_cfg = {}
    cfg_file = yaml_load(cfg_str)
    if cfg_path is not None:
        abs_cfg_path = os.path.abspath(cfg_path)
        if abs_cfg_path in seen_cfg:
            if seen_cfg[abs_cfg_path] is None:
                raise RuntimeError("Circular reference detected in config file")
            else:
                return copy.deepcopy(seen_cfg[abs_cfg_path])
        seen_cfg[abs_cfg_path] = None
    if isinstance(cfg_file, dict):
        cfg_file = parse_config(
            cfg_file, cfg_path=cfg_path, seen_cfg=seen_cfg, **kwargs
        )
    if cfg_path is not None:
        seen_cfg[abs_cfg_path] = cfg_file
    return cfg_file


def yaml_with_merge_load(stream, path=None, ext_vars=None):
    config = parse_str(stream, path=path)
    if ext_vars is not None and isinstance(ext_vars, dict) and isinstance(config, dict):
        config = deep_update(config, ext_vars)
    return config


set_loader("yaml_with_merge", yaml_with_merge_load)
set_dumper("yaml_with_merge", dumpers["yaml"])
