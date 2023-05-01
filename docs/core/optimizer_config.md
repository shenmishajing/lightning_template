## Optimizer config

[lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html) only support [one optimizer](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli_intermediate_2.html#multiple-optimizers) and [at most one lr scheduler](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli_intermediate_2.html#multiple-schedulers) using `--optimizer` and `--lr_scheduler` flags, which may can not satisfy our need sometimes.

Therefore, we add a new flag named `--optimizer_config` to support more complex optimizer configuration. The value of `--optimizer_config` flag is a very complex object, let's describe it step by step.

## Overview

First, we display the complete config object here, so you can get the whole picture, and jump back to here anytime you reading the following context.

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

As described in [Overview](#overview), value of `--optimizer_config` flag is a very complex object, let's describe it level by level. First,the value should be a single `optimize_config` dict or a list of `optimize_config` dict, a single `optimize_config` dict is equal to a list with only one item which is a `optimize_config` dict.

```yaml
optimizer_config:
    <a single optimize_config object>
optimizer_config:
    -   <a single optimize_config object>
    -   <a single optimize_config object>
    -   <a single optimize_config object>
```

A `optimize_config` dict can contains three keys, which are `optimizer` `frequency` and `lr_scheduler`, with values `<a optimizer config object>` `<null or int>` and `<a lightning lr scheduler config object>`.

```yaml
# optimize_config object
optimizer:
    <a optimizer config object>
frequency: <null or int>
lr_scheduler:
    <a lightning lr scheduler config object>
```

The `frequency` and `lr_scheduler` key is optional, so `<a optimizer config object>` will also can be put here and it will be warppered as `{'optimizer': <a optimizer config object> }`. The following `optimize_config` dict

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

The `frequency` key can only used when there are multi optimizers, and it have to neither set to None for all optimizers or set to int for all optimizers, it will raise an error if some optimizers set to None and others set to int.

#### `frequency` key is None

When all `frequency` set to None, every optimizers will be used to update model on every batch.

#### `frequency` key is int

When all `frequency` set to int, every batch will select only one optimizer according to the batch idx, and only the selected one will be used to update model on this batch.

For example, if there are two optimizers with `frequency` equal to 2 and 3 respectively. On every 5 batches, the first 2 batches will select the first optimizer and use it only to update the model, the last 3 batches will select the second optimizer and use it only to update the model. Therefore, set all `frequecy` to None is not equal to set all of them to `1`.

## Optimizer config

`<a optimizer config object>` represent a optimizer following lightning CLI instantiate_class arguments format, which means it contains two keys named `class_path` and `init_args`. `class_path` is a import str to the class, `init_args` is optional, if exist, its value will be used to instantiate the class. For more details, see [arguments with class type doc](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli_advanced_3.html#trainer-callbacks-and-arguments-with-class-type).

But, there is no way to specific model's parameters for optimizer, especially when there are many optimizers. Therefore, we add a method to support this. Every arguments in `init_args` work trivially, but the params where should be model's parameters work differently. We use a str or None or List[str, None] to represent a list of model's parameters. a str represent a list of model's parameters with name startswith this str, but if a parameter will only appear once, so if there are multi str match the same parameter, this parametr will be matched by the longest str. If some parameters do not be matched by any str, it will be matched by None.

For example, if a model has a fc layer and a backbone which contains layer 0-3. The following optimizer config

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

will construct a optimizer with three params groups as follow:

```yaml
-   [ backbone.layer0, backbone.layer3 ]
-   [ backbone.layer1, backbone.layer2 ]
-   [ fc ]
```

## Lightning lr scheduler config

`<a lightning lr scheduler config object>` represent a lightning lr scheduler, which contains several keys named `scheduler`, `interval`, `frequency`, etc. All keys other than `scheduler` are optional, and their default value as following, for more details, see [configure optimizers doc](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#configure-optimizers):

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

In fact, `<a lightning lr scheduler config object>` also contains `opt_idx` and `reduce_on_plateau` keys, but lightning will set them automatically, so we do not need to set them maunally.

## lr scheduler config

As [optimizer config](#optimizer-config), we use lightning CLI instantiate_class arguments format to represent a lr scheduler. The optimizer argument of it will be set to the `optimizer` in the same `optim_config object`, so there is no need to set it maunally.

For example, if a typical lr scheduler config will look like:

```yaml
# lr scheduler config object
class_path: torch.optim.lr_scheduler.MultiStepLR
init_args:
    milestones: [8, 11]
```

## warmup lr scheduler config

A warmup lr scheduler (implemented by [this method](model.md#manual-lr-scheduler)) config is a part-support Lightning lr scheduler config, which means only the scheduler and frequency key are supported, the `interval` will be set to `step` forcefully.

A complete warmup lr scheduler config will look like:

```yaml
# warmup lr scheduler config object
warmup_config:
    scheduler:
        class_path: lightning_template.utils.optim.WarmupScheduler
        init_args:
            warmup_iters: 500
    frequency: 1
```

As the `frequecy` is optional, you omit it, and use a [lr scheduler config](#lr-scheduler-config) as warmup lr scheduler config. Therefore, it will look like:

```yaml
# warmup lr scheduler config object
warmup_config:
    class_path: lightning_template.utils.optim.WarmupScheduler
    init_args:
        warmup_iters: 500
```

Furthermore, if you use the `utils.optim.WarmupScheduler` as warmup scheduler, you can omit it also, now the warmup scheduler config will look like:

```yaml
# warmup lr scheduler config object
warmup_config:
    warmup_iters: 500
```

For more detail of `utils.optim.WarmupScheduler`, see [github source](https://github.com/shenmishajing/project_template/blob/main/utils/optim/warmup_lr_scheduler.py)
