# Usage

## Toy project

The faster way to develop a new project based on [project template](https://github.com/shenmishajing/project_template) is to try to write a toy project based on it. You can refer to [toy project](https://github.com/shenmishajing/toy_project) on how to implement a new project.

## Create models and datasets

Similar to [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/), we use LightningModule to implement the model and train, val, and test loop, and use LightningDataModule to implement dataset and dataloaders, for details, see [model doc](../core/model.md) and [dataset doc](../core/dataset.md)

### Dataset

We recommend you start a new project from the dataset since you can debug your dataset without a model but the reverse is not true.

There is no tricky thing for the implementation of datasets, so just implement your datasets as usual. The tricky thing happens when you want to configure your dataset from a config file. You can refer to the [toy dataset config file]( https://github.com/shenmishajing/toy_project/blob/main/configs/datasets/toy_dataset/toy_dataset.yaml) for more details.

You can copy most configuration in that file to your dataset configuration file, and then you can modify it based on your dataset.

From that config file, we can learn several things:

- Firstly, we use a dict with the key `class_path` and `init_args` to init an object from a class. The `class_path` is the import path of the class, and the `init_args` include the arguments to init the object. You can refer to the [doc](https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes) of [jsonargparse](https://github.com/omni-us/jsonargparse/) for more details.
- Secondly, we can use the `__base__` to import another config file with a relative path. You can refer to the [doc](../configs/argument_parsers/yaml_with_merge.md) for more details.
- Thirdly, we can use a dict with split names (`fit`, `val` etc.) as the key and the configuration dict as the value to configure both datasets and dataloaders. We also can utilize the `split_info` configuration to reduce the redundancy of the configuration. For example, we can use the `split_info` to set the `subset` argument as the value in `split_name_map`. You can refer to the [doc](../core/dataset.md) for more details.
- Finally, we can have multiple datasets for a split, the configuration of which will inherit one by one, and the configuration of the next split will inherit the configuration of the first dataset in the previous split. You can refer to the [doc](../core/dataset.md) for more details.

### Model

#### Model definition

We recommend you define your model in the `build_model` method, which will be called by the `configure_model` hook of [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/).

The `configure_model` hook will speed up the initialization of your model and reduce the amount of CPU memory required, especially when you are using shared strategies, such as DeepSpeed and FSDP. However, the `configure_model` method will be called during each of fit/val/test/predict stages in the same process, so ensure that implementation of this hook is idempotent, i.e., after the first time the hook is called, subsequent calls to it should be a no-op (see the [doc](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-model) from [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) for more details), which is error prone. Therefore, we design the `build_model` method to avoid the most common errors, with which you can init your model as in the `__init__` method. You can refer to the implementation of the [toy model python file](https://github.com/shenmishajing/toy_project/blob/main/src/project/models/toy_model.py) for more details.

#### Forward method

The key interface of the model is the `forward` method, you can refer to the [toy model config file](https://github.com/shenmishajing/toy_project/blob/main/src/project/models/toy_model.py) for the output structure of the `forward` method.

##### Losses for training

The output of the `forward` method should be a dict with the key `loss_dict` which contains all the loss of the model. The total loss will be calculated automatically as the sum of all the values in the `loss_dict` with `loss` in key name. If you want to calculate the total loss in a different way, you have to set the `loss` key in the `loss_dict` to  the total loss you want manually.

##### Metrics for evaluation

If you want to calculate metrics for your model, the output of the `forward` method should also contain a dict with the key `metric_dict` which contains the input arguments to calculate the metrics. We recommend you use the metrics implemented in [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/). If so, you only need to return a `metric_dict` with `preds` and `target` key as [toy model python file](https://github.com/shenmishajing/toy_project/blob/main/src/project/models/toy_model.py) and set the metrics you want to use like [metrics config file](https://github.com/shenmishajing/toy_project/blob/main/configs/metrics/classification.yaml).

Both the losses and the metrics will be logged automatically.

#### Prediction and visualization

If you want to visualize the prediction of your model, you can implement a series of tasks func with name `predict_<task-name>` in the `LightningModule` and set the `predict_tasks` argument to the list of the name of tasks you want to run. The `predict_forward` func will be called before the prediction to prepare some results which should be shared between all the prediction tasks, and all the `predict_<task-name>` func mentioned in `predict_tasks` will be called to visualize the prediction. You can refer to the [doc](../core/model.md) and the [toy model python file](https://github.com/shenmishajing/toy_project/blob/main/src/project/models/toy_model.py) for more details.

By default, if you use the `predict_path` and `output_path` as the path to save your prediction or visualization result. You can find your result in `prediction` folder in your working directory if you do not set the `ckpt_path`. If you set the `ckpt_path` to the checkpoint you want to use, you can find your result in the `prediction` folder in the parent folder of the checkpoint.

#### The model argument

Note that we recommend you inherit the `LightningModule` directly, which will facilitate the development of your model. However, if you have to use the model implemented in other libraries, you can import it as the `model` argument of the `LightningModule` and implement the `forward` method to wrap the model.

## Config optimizers and lr schedulers

[pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) does not support multi optimizers and lr schedulers from cli, we add this feature, see [doc](../core/optimizer_config.md) for detail.

## Cross-validation

Set `num_folds` of the trainer to an int bigger than one to start cross-validation, for details, see [doc](../core/trainer.md).

## Config files

See [config file structure](../configs/config_file_structure.md), [deep update](../configs/deep_update.md), [yaml with merge](../configs/argument_parsers/yaml_with_merge.md), and [json file](../configs/argument_parsers/json_file.md) for more details.

You can get some examples from [project_template](https://github.com/shenmishajing/project_template) and [toy project](https://github.com/shenmishajing/toy_project).

## CLI

This project is based on the [lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html), so it supports all features from [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) and [lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html), you can get a brief introduction from [cli doc](../tools/cli.md).

Briefly, use the following command to train, validate, test, or predict your model.

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> cli {fit, validation, test, predict} --config configs/runs/path/to/config [ --ckpt_path path/to/checkpoint ]
```

The `gpu_ids` is a comma-separated id list or just one int. The `ckpt_path` is an optional path to the checkpoint you want to use for resume fit, validation, test, or prediction.

## Wandb Logger

We support logging your code with `WandbNamedLogger` to control the version of your codes for every experiment. To use `Wandb`, you have to create an account on their [site](https://wandb.ai/) and login following their [doc](https://docs.wandb.ai/quickstart).
