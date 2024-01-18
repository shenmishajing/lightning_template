# Usage

## Create models and datasets

Similar to [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/), we use LightningModule to implement the model and train, val, and test loop, and use LightningDataModule to implement dataset and dataloaders, for details, see [model doc](../core/model.md) and [dataset doc](../core/dataset.md)

## Config files

### Config optimizers and lr schedulers

[pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) does not support multi optimizers and lr schedulers from cli, we add this feature, see [doc](../core/optimizer_config.md) for detail.

### Cross-validation

Set `num_folds` of the trainer to an int bigger than one to start cross-validation, for details, see [doc](../core/trainer.md).

See [config file structure](../configs/config_file_structure.md), [deep update](../configs/deep_update.md), [yaml with merge](../configs/argument_parsers/yaml_with_merge.md), and [json file](../configs/argument_parsers/json_file.md) for more details.

You can get some examples from [project_template](https://github.com/shenmishajing/project_template)

## CLI

This project is based on the [lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html), so it supports all features from [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) and [lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html), you can get a brief introduction from [cli doc](../tools/cli.md).

## Wandb Logger

We support logging your code with `WandbNamedLogger` to control the version of your codes for every experiment. To use `Wandb`, you have to create an account on their [site](https://wandb.ai/) and login following their [doc](https://docs.wandb.ai/quickstart).
