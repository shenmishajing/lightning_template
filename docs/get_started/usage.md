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

We support logging your code with `WandbNamedLogger` to control the version of your codes for every experiment. To use `Wandb`, you have to create an account on their [site](https://wandb.ai/) and login following their [doc](https://docs.wandb.ai/quickstart). We recommend you symlink every file you want to log under the `code_dirs` folder of your project since `WandbNamedLogger` has to walk through that folder to find the files you want to log. You can use the following command to do that:

```bash
mkdir code_dirs
cd code_dirs
ln -s ../* ../.* .
cd ..
```

Note that you should run this command in the root folder of your project after you clone this project and you have to symlink any new files and folders you want to log to wandb manually after that. You should not link any files or folders you don't want to log under the `code_dirs` folder. If you do not create the `code_dirs` folder, `WandbNamedLogger` will walk through the root folder of your project to find the files you want to log, which will be slow, especially if you put your datasets with too many files and folders in your project. Therefore, we recommend you use the above commands to create the `code_dirs` folder.
