## Introduction

A generic project template lib based on [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) for [project-template](https://github.com/shenmishajing/project_template)

## Feature

- All features from [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) and [lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html). Especially, experiment manager feature, auto implement multi-node, multi-device, multi-accelerator support, etc.
- Powerful [deep update](docs/configs/deep_update.md) feature for config file inherit to manage your config files in a more hierarchical way, see also [recommand config file structure](docs/configs/config_file_structure.md).
- Multi and complex optimizers and lr_scheduler from CLI config support, see [doc](docs/core/optimizer_config.md).
- Powerful and flexible LightningModule and LightningDataModule base class.
- Cross-validation support with only one argument you have to change.
- Useful auto lr finder and auto batch size scaler.

## Installation

See [installation docs](docs/installation/installation.md) for details.

## Usage

### CLI

This project base on the [lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html), so it support all feature from [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) and [lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html), you can get a brief introduction from [cli doc](docs/tools/cli.md).

### Create models and datasets

As [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/), we use LightningModule to implement the model and train, val and test loop, use LightningDataModule to implement dataset and dataloaders, for detail, see [model doc](docs/core/model.md) and [dataset doc](docs/core/dataset.md)

### Config optimizers and lr schedulers

[pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) do not support multi optimizers and lr schedulers from cli, we add this feature, see [doc](docs/core/optimizer_config.md) for detail.

### Cross-validation

Set `num_folds` of trainer to a int bigger than one to start cross-validation, for details, see [doc](docs/core/trainer.md).

### Config files

See [config file structure](docs/configs/config_file_structure.md), [deep update](docs/configs/deep_update.md), [yaml with merge](docs/configs/argument_parsers/yaml_with_merge.md) and [json file](docs/configs/argument_parsers/json_file.md)

You can get some example from [project_template](https://github.com/shenmishajing/project_template)
