## Introduction

The base LightningModule to inherit.

## Arguments and config

The base LightningModule has two arguments named `model` and `loss_weights`. `model` is a normal `torch.nn.Module` to fit or predict etc., and `loss_weights` is a dict, if you use the `loss_step` method from the base LightningModule, the loss dict will multi the loss weight dict before calculate the total loss.

## Manual lr scheduler

When we use multi lr scheduler with one optimizer, typically one optimizer with a lr scheduler and a [warmup scheduler](optimizer_config.md#warmup-lr-scheduler-config) (the warmup scheduler described [here](optimizer_config.md#warmup-lr-scheduler-config) is implemented in this way), we will get in trouble with Lightning and lr monitor callback. Therefore, we support the manual lr scheduler, they are not a lightning lr scheduler, and is not known by Ligihtning, they are just called at their `frequency` after every `optimize_step`.

## Train loop

We use `forward` and `loss_step` to calculate loss for train loop. By default, `forward` method of base `LightningModule` will the call the `forward` method of `model` attr, and `loss_step` method will return the result of `forward` method. Therefore, if you want to implement your own logic and reuse our train step code in the same time, you may need to overwrite the `forward` and `_loss_step` method.

## Valiadiot and Test loop

We use `forward_step` to calculate the metrics and `on_forward_epoch_end` method to accumulate the metrics for val and test loop. By default, we calculate loss in `forward_step` as we do in train loop, and do nothing in `on_forward_epoch_end`. Therefore, if you want to implement your own logic and reuse our train step code in the same time, you may need to overwrite the `forward_step` and `on_forward_epoch_end` method.

Note that, after `lightning` 2.0, `lightning` change the method `forward_epoch_end` to `on_forward_epoch_end`, and do not catch the results from `forward_step` method to put them as a list to `forward_epoch_end` method. Therefore, we have to use other method to accumulate the metrics, for example, you may need to use a list as a var of the `LightningModule` to save the result manually.

## Predict loop

We use `predict_tasks` attr to identify the predict task we have to do during prediction. The `LightningModule` accept a list of str as task names, for each name, we create the output_path for saving visualization results, and save them as key-value pair in `predict_taks` dict. During predict loop, for each key in `predict_tasks` dict, we call `predict_{key-name}` method to start task in prediction loop, with arguments from `predict_forward` method and `output_path` = `{value}`.
