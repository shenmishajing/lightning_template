```{toctree}
:hidden:
:caption: Installation

installation/installation
```

```{toctree}
:hidden:
:caption: Core

core/model
core/dataset
core/optimizer_config
core/trainer
```

```{toctree}
:hidden:
:caption: Configuration

configs/deep_update
configs/config_file_structure
configs/argument_parsers/index
```

```{toctree}
:hidden:
:caption: Tools

tools/cli
```

<!-- ```{toctree}
:hidden:
:caption: API Referrences

api/index
``` -->

# Lightning Template

## Introduction

A generic project template lib based on [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) for [project-template](https://github.com/shenmishajing/project_template)

## Feature

- All features from [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) and [lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html). Especially, the experiment manager feature, auto-implementing multi-node, multi-device, multi-accelerator support, etc.
- Powerful [deep update](configs/deep_update.md) feature for config file inherit to manage your config files more hierarchically, see also [recommend structure](configs/config_file_structure.md).
- Multi and complex optimizers and lr_scheduler from CLI config support, see [doc](core/optimizer_config.md).
- Cross-validation support with only one argument you have to change, see [doc](core/trainer.md).
- Hyperparameters tuning via Wandb, see the [doc](configs/argument_parsers/json_file.md) and [wandb sweep](https://docs.wandb.ai/guides/sweeps).
- Powerful and flexible LightningModule and LightningDataModule base class.
- Useful auto lr finder and auto batch size scaler, see [doc](tools/cli.md).
- Useful tools to compare the speed of different models, see [speed-benchmark lib](https://github.com/shenmishajing/speed_benchmark)
- Useful command line scripts launcher, see [shell-command-launcher lib](https://github.com/shenmishajing/shell_command_launcher)

## Installation

See [installation docs](installation/installation.md) for details.

## Usage

### CLI
This project is based on the [lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html), so it supports all features from [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) and [lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html), you can get a brief introduction from [cli doc](tools/cli.md).

### Create models and datasets

Similar to [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/), we use LightningModule to implement the model and train, val, and test loop, and use LightningDataModule to implement dataset and dataloaders, for details, see [model doc](core/model.md) and [dataset doc](core/dataset.md)

### Config optimizers and lr schedulers

[pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) does not support multi optimizers and lr schedulers from cli, we add this feature, see [doc](core/optimizer_config.md) for detail.

### Cross-validation

Set `num_folds` of the trainer to an int bigger than one to start cross-validation, for details, see [doc](core/trainer.md).

### Config files

See [config file structure](configs/config_file_structure.md), [deep update](configs/deep_update.md), [yaml with merge](configs/argument_parsers/yaml_with_merge.md), and [json file](configs/argument_parsers/json_file.md).

You can get some examples from [project_template](https://github.com/shenmishajing/project_template)

### Speed Benchmark

See [speed-benchmark lib](https://github.com/shenmishajing/speed_benchmark)

### Shell Command Launcher

See [shell-command-launcher lib](https://github.com/shenmishajing/shell_command_launcher)
