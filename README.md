## Introduction

A generic project template lib based on [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) for [project-template](https://github.com/shenmishajing/project_template)

## Feature

- All features from [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) and [lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html). Especially, the experiment manager feature, auto-implementing multi-node, multi-device, multi-accelerator support, etc.
- Powerful [deep update](docs/configs/deep_update.md) feature for config file inherit to manage your config files more hierarchically, see also [recommend structure](docs/configs/config_file_structure.md).
- Multi and complex optimizers and lr_scheduler from CLI config support, see [doc](docs/core/optimizer_config.md).
- Cross-validation support with only one argument you have to change, see [doc](docs/core/trainer.md).
- Hyperparameters tuning via Wandb, see the [doc](docs/configs/argument_parsers/json_file.md) and [wandb sweep](https://docs.wandb.ai/guides/sweeps).
- Powerful and flexible LightningModule and LightningDataModule base class.
- Useful auto lr finder and auto batch size scaler, see [doc](docs/tools/cli.md).

## Installation

See [installation docs](docs/installation/installation.md) for details.

## Usage

### CLI
This project is based on the [lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html), so it supports all features from [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) and [lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html), you can get a brief introduction from [cli doc](docs/tools/cli.md).

### Create models and datasets

Like [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/), we use LightningModule to implement the model and train, val, and test loop, and use LightningDataModule to implement dataset and dataloaders, for detail, see [model doc](docs/core/model.md) and [dataset doc](docs/core/dataset.md)

### Config optimizers and lr schedulers

[pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) does not support multi optimizers and lr schedulers from cli, we add this feature, see [doc](docs/core/optimizer_config.md) for detail.

### Cross-validation

Set `num_folds` of the trainer to an int bigger than one to start cross-validation, for details, see [doc](docs/core/trainer.md).

### Config files

See [config file structure](docs/configs/config_file_structure.md), [deep update](docs/configs/deep_update.md), [yaml [with merge](docs/configs/argument_parsers/yaml_with_merge.md), and [json file](docs/configs/argument_parsers/json_file.md)](docs/configs/argument_parsers/json_file.md)

You can get some examples from [project_template](https://github.com/shenmishajing/project_template)

### Speed Benchmark

See [speed benchmark lib](https://github.com/shenmishajing/speed_benchmark)
