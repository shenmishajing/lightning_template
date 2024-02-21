import os
import re
from functools import partial
from glob import iglob
from sys import version_info
from typing import Optional, Union

import wandb
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.loggers.wandb import WandbLogger


class SetWandbLoggerCallback(Callback):
    def __init__(
        self,
        log_code_cfg: Optional[Union[dict, bool]] = None,
        watch_model_cfg: Optional[Union[dict, bool]] = None,
    ):
        self.log_code_cfg = log_code_cfg if log_code_cfg is not None else {}
        self.watch_model_cfg = watch_model_cfg if watch_model_cfg is not None else {}

        if isinstance(self.log_code_cfg, dict):
            self.log_code_cfg.setdefault("name", "code")
            default_log_code_cfg = {
                "include_dirs": ["src", "tests", "tools", "docs", "configs", ".github"],
                "include_file_types": [
                    "py",
                    "yaml",
                    "yml",
                    "toml",
                    "sh",
                    "md",
                    "gitignore",
                    "ini",
                ],
                "exclude_patterns": ["__pycache__"],
            }
            for key, value in default_log_code_cfg.items():
                if key not in self.log_code_cfg:
                    self.log_code_cfg[key] = value
                else:
                    if not isinstance(self.log_code_cfg[key], list):
                        self.log_code_cfg[key] = [self.log_code_cfg[key]]
                    self.log_code_cfg[key].extend(value)

            include_patterns = []
            for type in self.log_code_cfg["include_file_types"]:
                for dir_name in self.log_code_cfg["include_dirs"]:
                    include_patterns.append(f"{dir_name}/**/*.{type}")
                include_patterns.append(f"*.{type}")
            include_patterns.extend(["LICENSE*", "license*"])

            self.log_code_cfg["exclude_patterns"] = [
                re.compile(p) for p in self.log_code_cfg["exclude_patterns"]
            ]

            def _is_match(path, patterns=None):
                return any(p.search(path) is not None for p in patterns)

            self.log_code_cfg["include_patterns"] = include_patterns
            self.log_code_cfg["exclude_fn"] = partial(
                _is_match, patterns=self.log_code_cfg["exclude_patterns"]
            )

    def log_code(self, trainer: Trainer):
        art = wandb.Artifact(self.log_code_cfg["name"], "code")
        files_added = False
        for glob_pattern in self.log_code_cfg["include_patterns"]:
            glob_params = {}
            if self.log_code_cfg.get("root") is not None:
                glob_params["root_dir"] = self.log_code_cfg["root"]
            if version_info >= (3, 11):
                glob_params["include_hidden"] = True
            for file_path in iglob(glob_pattern, recursive=True, **glob_params):
                if not self.log_code_cfg["exclude_fn"](file_path):
                    files_added = True
                    save_name = (
                        file_path
                        if self.log_code_cfg.get("root") is None
                        else os.path.relpath(file_path, self.log_code_cfg["root"])
                    )
                    art.add_file(file_path, name=save_name)
        if not files_added:
            wandb.termwarn(
                "No relevant files were detected in the specified directory. No code will be logged to your run."
            )
            return None

        return trainer.logger.experiment.log_artifact(art)

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None
    ) -> None:
        if trainer.logger is not None and isinstance(trainer.logger, WandbLogger):
            if isinstance(self.watch_model_cfg, dict):
                trainer.logger.watch(model=pl_module, **self.watch_model_cfg)
            if isinstance(self.log_code_cfg, dict):
                self.log_code(trainer)
