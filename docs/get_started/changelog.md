# Changelog

## 1.6.6 (2024-01-27)

### Fix

- **predict_path**: fix hardcode predict_path bug

## 1.6.5 (2024-01-26)

### Fix

- **get_cfg_from_path**: fix get_cfg_from_path change dir bug
- **get_split_config**: fix get_split_config last config bug

## 1.6.4 (2024-01-23)

### Refactor

- **update_evaluator**: add metrics param to update_evaluator

## 1.6.3 (2024-01-23)

### Fix

- **SetWandbLoggerCallback**: fix root_dir bug

## 1.6.2 (2024-01-23)

### Fix

- **SetWandbLoggerCallback**: fix SetWandbLoggerCallback does not support 3.8-3.10 bug

## 1.6.1 (2024-01-22)

### Fix

- **checkpoint**: fix do not save last checkpoint if it is not topk best checkpoint

## 1.6.0 (2024-01-20)

### Feat

- **update_evaluator**: add split arg for update_evaluator func

### Fix

- **model.loss_step**: use the same interface for model func
- **on_forward_epoch_end**: fix arg order bug

## 1.5.5 (2024-01-18)

### Fix

- **wandb-log-code**: fix use code_dir does not follow link bug, use glob instead of walk

### Refactor

- **log_dir**: do not set log_dir in before_instantiate_classes

## 1.5.4 (2024-01-16)

### Refactor

- **SaveAndLogConfigCallback**: move log_hyperparams to save_config func

## 1.5.3 (2024-01-16)

### Fix

- **modelcheckpoint**: fix link bug for best checkpoint

## 1.5.2 (2024-01-16)

### Fix

- **progress**: fix divide zero bug

## 1.5.1 (2024-01-15)

### Fix

- **log-code**: fix log code include files bug for wandb logger

## 1.5.0 (2024-01-14)

### Feat

- **eval-metrics**: only update evaluators when metric dicts is set explicitly
- **split-name**: add feature to customize split names

## 1.4.12 (2024-01-10)

### Fix

- **LightningModule**: fix output is not dict bug

## 1.4.11 (2024-01-10)

### Fix

- **ActionJsonFile**: fix multi ActionConfigFile bug
- **LightningModule**: fix recursive_parse_modules is not staticmethod bug

## 1.4.10 (2024-01-04)

### Fix

- **ci**: add write permission to contents for release ci

## 1.4.9 (2024-01-04)

### Fix

- **ci**: fix publish twice on tagged commit bug
- **ci**: fix check commit messages protect master br bug

## 1.4.8 (2024-01-04)

### Fix

- **ci**: fix fetch depth = 0 bug in check pre commit hooks

## 1.4.7 (2024-01-03)

### Fix

- **pyproject.commitizen**: use annotated tags instead of light weighted tags

## 1.4.6 (2024-01-03)

### Fix

- **parsers**: fix using non-public api of jsonargparse bug

## 1.4.5 (2024-01-03)

### Fix

- **changelog**: fix changelog name bug, add title for changelog

## 1.4.4 (2024-01-03)

### Fix

- **changelog**: move changelog.md to docs/get_started

## 1.4.3 (2024-01-03)

### Fix

- **commitizen**: add config for commitizen

## 1.4.2 (2023-12-28)

## 1.4.1 (2023-12-28)
