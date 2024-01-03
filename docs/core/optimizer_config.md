# Optimizer Config

[lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html) only supports [one optimizer](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli_intermediate_2.html#multiple-optimizers) and [at most one lr scheduler](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli_intermediate_2.html#multiple-schedulers) using `--optimizer` and `--lr_scheduler` flags, which may not satisfy our needs sometimes.

Therefore, we added a new flag named `--optimizer_config` to support more complex optimizer configurations. The value of `--optimizer_config` flag is a very complex object, let's describe it step by step.

## Overview

First, we display the complete config object here, so you can get the whole picture, and jump back to here anytime you read the following context.

```yaml
optimizer_config:
    -   optimizer:
            class_path: torch.optim.AdamW
            init_args:
                params:
                    -   params: backbone
                        lr: 1e-4
                    -   params: [ backbone.layer1, backbone.layer2 ]
                        weight_decay: 1e-4
                    -   null
                lr: 1e-3
                weight_decay: 1e-2
        frequency: null
        lr_scheduler:
            scheduler:
                class_path: torch.optim.lr_scheduler.MultiStepLR
                init_args:
                    milestones: [8, 11]
            interval: epoch
            frequency: 1
            monitor: val_loss
            strict: True
            name: None
            warmup_config:
                scheduler:
                    class_path: lightning_template.utils.optim.WarmupScheduler
                    init_args:
                        warmup_iters: 500
                frequency: 1
    -   optimizer:
            class_path: torch.optim.AdamW
            init_args:
                params:
                    -   params: backbone
                        lr: 1e-4
                    -   params: [ backbone.layer1, backbone.layer2 ]
                        weight_decay: 1e-4
                    -   null
                lr: 1e-3
                weight_decay: 1e-2
        frequency: null
        lr_scheduler:
            scheduler:
                class_path: torch.optim.lr_scheduler.MultiStepLR
                init_args:
                    milestones: [8, 11]
            interval: epoch
            frequency: 1
            monitor: val_loss
            strict: True
            name: None
            warmup_config:
                scheduler:
                    class_path: lightning_template.utils.optim.WarmupScheduler
                    init_args:
                        warmup_iters: 500
                frequency: 1
```

## Single optimize config

As described in [Overview](#overview), the value of `--optimizer_config` flag is a very complex object, let's describe it level by level. First of all, the value should be a single `optimize_config` dict or a list of `optimize_config` dict, a single `optimize_config` dict is equal to a list with only one item which is a `optimize_config` dict.

```yaml
optimizer_config:
    <a single optimize_config object>
optimizer_config:
    -   <a single optimize_config object>
    -   <a single optimize_config object>
    -   <a single optimize_config object>
```

A `optimize_config` dict can contain three keys, which are `optimizer` `frequency` and `lr_scheduler`, with values `<a optimizer config object>` `<null or int>` and `<a lightning lr scheduler config object>`.

```yaml
# optimize_config object
optimizer:
    <a optimizer config object>
frequency: <null or int>
lr_scheduler:
    <a lightning lr scheduler config object>
```

The `frequency` and `lr_scheduler` key are optional, so `<a optimizer config object>` can also be put here and it will be parsed as `{'optimizer': <an optimizer config object> }`, which means that the following `optimize_config` dict

```yaml
# optimize_config object
<a optimizer config object>
```

will be treated as

```yaml
# optimize_config object
optimizer:
    <a optimizer config object>
```

### `frequency` key

The `frequency` key can only used when there are multiple optimizers, and it has to be either set to None for all optimizers or set to int for all optimizers, it will raise an error if the values of `frequency` of some optimizers have been set to None and others have been set to int.

#### `frequency` key is None

When all `frequency` are set to None, every optimizer will be used to update the model on every iteration.

#### `frequency` key is int

When all `frequency` are set to int, only one optimizer will be selected to update the model on every iteration according to the batch index.

For example, if there are two optimizers with `frequency` equal to 2 and 3 respectively. On every 5 batches, the first optimizer will selected on the first 2 batches and the second optimizer will be selected on the last 3 batches. For every batch, only the selected optimizer will be used to update the model. Therefore, setting all `frequecy` to None is not equal to setting all of them to `1`.

## Optimizer config

`<a optimizer config object>` represents an optimizer following lightning CLI instantiate_class arguments format, which means it contains two keys named `class_path` and `init_args`. `class_path` is an import str to the class, `init_args` is optional, if exists its value will be used to instantiate the class. For more details, see [arguments with class type doc](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli_advanced_3.html#trainer-callbacks-and-arguments-with-class-type).

However, there is no way to specify the parameters for optimizers in the lightning CLI instantiate_class arguments format, especially when there are many optimizers. Therefore, we add a method to support this. We use a `str` or `None` or `List[str, None]` to represent a list of the model's parameters. a `str` represents a list of the model's parameters with a name starts with this str, but a parameter will only appear once, so if multiple strs match the same parameter, this parameter will be matched by the longest str. If some parameters are not matched by any str, it will be matched by None.

For example, if a model has a fc layer and a backbone that contains layers 0-3. The following optimizer config

```yaml
# optimizer config object
class_path: torch.optim.AdamW
init_args:
    params:
        -   params: backbone
            lr: 1e-4
        -   params: [ backbone.layer1, backbone.layer2 ]
            weight_decay: 1e-4
        -   null
    lr: 1e-3
    weight_decay: 1e-2
```

will construct an optimizer with three params groups as follows:

```yaml
-   [ backbone.layer0, backbone.layer3 ]
-   [ backbone.layer1, backbone.layer2 ]
-   [ fc ]
```

## Lightning lr scheduler config

`<a lightning lr scheduler config object>` represents a lightning lr scheduler, which contains several keys named `scheduler`, `interval`, `frequency`, etc. All keys other than `scheduler` are optional, and their default value is as follows, for more details, see [configure optimizers doc](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#configure-optimizers):

```yaml
# lightning lr scheduler config object
lr_scheduler:
    scheduler:
        <a lr scheduler config object>
    interval: epoch
    frequency: 1
    monitor: val_loss
    strict: True
    name: null
    warmup_config:
        <a warmup lr scheduler config object>
```

In fact, `<a lightning lr scheduler config object>` also contains `opt_idx` and `reduce_on_plateau` keys, but lightning will set them automatically, so we do not need to set them manually.

## lr scheduler config

For the [optimizer config](#optimizer-config), we use lightning CLI instantiate_class arguments format to represent a lr scheduler. The optimizer argument will be set to the `optimizer` in the same `optim_config object`, so there is no need to set it manually.

For example, a typical lr scheduler config will look like this:

```yaml
# lr scheduler config object
class_path: torch.optim.lr_scheduler.MultiStepLR
init_args:
    milestones: [8, 11]
```

## warmup lr scheduler config

A warmup lr scheduler (implemented by [this method](model.md#manual-lr-scheduler)) config is a part-support Lightning lr scheduler config, which means only the scheduler and frequency key are supported, and the `interval` will be set to `step` forcefully.

A complete warmup lr scheduler config will look like this:

```yaml
# warmup lr scheduler config object
warmup_config:
    scheduler:
        class_path: lightning_template.utils.optim.WarmupScheduler
        init_args:
            warmup_iters: 500
    frequency: 1
```

Since the `frequecy` is optional, you can omit it, and use a [lr scheduler config](#lr-scheduler-config) as a warmup lr scheduler config. Therefore, it will look like:

```yaml
# warmup lr scheduler config object
warmup_config:
    class_path: lightning_template.utils.optim.WarmupScheduler
    init_args:
        warmup_iters: 500
```

Furthermore, if you use the `lightning_template.utils.optim.WarmupScheduler` as warmup scheduler, you can omit it also, now the warmup scheduler config will look like this:

```yaml
# warmup lr scheduler config object
warmup_config:
    warmup_iters: 500
```

For more detail of `lightning_template.utils.optim.WarmupScheduler`, see the [source code](https://github.com/shenmishajing/project_template/blob/main/utils/optim/warmup_lr_scheduler.py)
