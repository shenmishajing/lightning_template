## Introduction

The base LightningModule to inherit.

## Arguments and config

The base LightningModule has two arguments named `model` and `loss_weights`. `model` is a normal `torch.nn.Module` to fit or predict etc., and `loss_weights` is a dict, if you use the `loss_step` method from the base LightningModule, the loss dict will multi the loss weight dict before calculate the total loss.

## Manual lr scheduler

When we use multi lr scheduler with one optimizer, typically one optimizer with a lr scheduler and a [warmup scheduler](optimizer_config.md#warmup-lr-scheduler-config) (the warmup scheduler described [here](optimizer_config.md#warmup-lr-scheduler-config) is implemented in this way), we will get in trouble with Lightning and lr monitor callback. Therefore, we support the manual lr scheduler, they are not a lightning lr scheduler, and is not known by Ligihtning, they are just called at their `frequency` after every `optimize_step`.

## Train loop

We use `forward` and `loss_step` to calculate loss for train loop by default, so if you want to use our train step code, you may need to implement the `forward` and `_loss_step` method.

## Valiadiot and Test loop

We use `forward_step` and `forward_epoch_end` method to calculate loss for train loop by default, so if you want to use our train step code, you may need to implement the `forward` and `_loss_step` method.

## Predict loop

We use `_output_paths` attr to identify the visualization task we have to do during prediction. For each name in `_output_paths` attr, we create the output_path according to the name for saving visualization results, and we call `{name}_visualization` to start every tasks.
