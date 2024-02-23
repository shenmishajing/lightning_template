# LightningModule

## Introduction

The base LightningModule to inherit.

## Arguments and config

The base LightningModule has several arguments and we will explain them one by one.

### model

The first argument is the `model`. We recommend you inherit the base LightningModule to implement your own model, but if you can not do this, you can pass your model as the `model` argument. The `forward` method of the base LightningModule will be passed to the `model` argument by default.

### ckpt_path

A list of path of checkpoints, if you pass this argument, the checkpoints will be loaded from the path and be used to init the model sequentially.

### evaluator_cfg

A dict to define your evaluators, which will be split for different split datasets. For more details, see the [Split attr set doc](dataset.md#split-attr-set) in the dataset doc. You can register all the evaluators as submodels of your model by setting the `evaluator_as_submodule` to `True`.

### finetune_cfg

A dict to define which parameters you want to finetune. The config would be like this:

```yaml
finetune_cfg:
    finetune: True
    params:
        - model.layer1
        - model.fc
        ...
```

If the `finetune` is set to `True`, the parameters in the `params` list will be set to `requires_grad=True`, otherwise, the parameters in the `params` list will be set to `requires_grad=False`. You can omit the `finetune` key, when you want to set it `True`, and you can only set the `finetune_cfg` to the list of params you want to finetune. Also, you use a single str instead of a list of str as params, which means only one group of parameters match that str will be selected.

### loss_weights

A dict for loss weights, if you use the `loss_step` method from the base LightningModule, the loss dict will multi the loss weight dict before calculating the total loss.

### predict_tasks

See the [Predict loop doc](#predict-loop) for details.

## Manual lr scheduler

When we use multi lr scheduler with one optimizer, typically one optimizer with a lr scheduler and a [warmup scheduler](optimizer_config.md#warmup-lr-scheduler-config) (the warmup scheduler described [here](optimizer_config.md#warmup-lr-scheduler-config) is implemented in this way), we will get in trouble with Lightning and lr monitor callback. Therefore, we support the manual lr scheduler, which is not a lightning lr scheduler and is not known by Ligihtning. Those schedulers are just called at their `frequency` after every `optimize_step`.

## Train, Validation and Test loop

### On iter

We call the `forward_step` in every iter, in which `forward` is called to get the outputs from the model, the `loss_step` is called to calculate the loss, and the `metric_step` is called to calculate the metrics. By default, the `forward` method of base `LightningModule` will call the `forward` method of `model` attr, the `loss_step` method will return the `loss_dict` in the outputs of the `forward` method and `metric_step` method will use the `metric_dict` in the outputs to update the evaluators.

### When epoch ends

We call the `on_forward_epoch_end` in which every epoch ends, in which `on_metric_epoch_end` is called to calculate the metrics for the model. By default, we calculate the metrics following the [TorchMetrics style](https://torchmetrics.readthedocs.io/en/stable/pages/overview.html). Therefore, if you use the [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/) as your evaluator, you do not need to write any code to calculate the metrics.

### Methods you need to override

Therefore, if you want to implement your own logic and reuse our train step code at the same time, you may need to return the `loss_dict` and the `metric_dict` in the `forward` method. And if your evaluators do not follow the interface as the TorchMetrics, you have to override the `update_evaluator` and `_compute_evaluator` method to adapt your own evaluator.

## Predict loop

We use the `predict_tasks` attr to identify the prediction tasks we have to complete during the prediction. The `LightningModule` accepts a list of str as task names, for each name, we create the output_path for saving visualization results, and save them as key-value pairs in the `predict_taks` dict. During the prediction loop, the `predict_forward` method will be called firstly to prepare some results which should be shared between all the prediction tasks. Then, for each key-value pair in the `predict_tasks` dict, we call `predict_{key}` method in the prediction, with arguments from the `predict_forward` method and `output_path` = `{value}`.
