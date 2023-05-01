## Interface

lightning-template provides several cli scripts for you to use. Generally, you can run all scripts under `lightning_tempalet/tools` by `python -m lightning_template.tools.<path.to.scripts>`, for example, use `python -m lightning_template.tools.cli` to run `tools/cli.py`. For convenience, we also provide some aliases for these scripts. All of them are as follows:

| alias             | scripts                                              |
| ----------------- | ---------------------------------------------------- |
| cli               | lightning_template/tools/cli.py                      |
| lr_finder         | lightning_template/tools/models/lr_finder.py         |
| batch_size_finder | lightning_template/tools/models/batch_size_finder.py |
| model_statistics  | lightning_template/tools/models/model_statistics.py  |

We recommmand you to use `alias` method to luanch scripts in terminal, and use `module` method to debug scripts in `vscode` etc.

For commands and options, you can get all available options and commands with `help` arguement, for example `cli --help`.

## CLI

### Train

You can start a experiment with command using `alias` as follow, in which, `gpu_ids` is comma-separated id list or just one int.

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> cli fit --config configs/runs/path/to/config
```

Samely, in `module` method.

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> python -m lightning_template.tools.cli fit --config configs/runs/path/to/config
```

## Validation Test and Predict

Just replace `fit` by `validation`, `test` or `predict` as follow:

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> python tools/cli.py {validation, test, predict} --config configs/runs/path/to/config
```

## Tune

### auto lr finder

Auto find the best learning rate for models, currently only support the first optimizer. You can get more information from [doc](https://lightning.ai/docs/pytorch/latest/advanced/training_tricks.html#learning-rate-finder).

You can use `lr_finder` alias or `lightning_template.tools.lr_finder` module to use this feature with this project.

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> lr_finder --config configs/runs/path/to/config
```

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python -m lightning_template.tools.models.lr_finder --config configs/runs/path/to/config
```

### auto scale batch size

Auto find the largest batch size or largest power of two as batch size. You can get more information from [doc](https://lightning.ai/docs/pytorch/latest/advanced/training_tricks.html#batch-size-finder).

Same as auto lr finder, you can use `batch_size_finder` alias or `lightning_template.tools.batch_size_finder` module to use this feature with this project.

Note that, the `batch_size_finder` script only support to run on only one device, which means even if you list many `gpu_ids` in `CUDA_VISIBLE_DEVICES` env var, it will only select the first one to run.

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> batch_size_finder --config configs/runs/path/to/config
```

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python -m lightning_template.tools.models.batch_size_finder --config configs/runs/path/to/config
```

## Model statistics

You can use `model_statistics` alias or `lightning_template.tools.model_statistics` module to calulate the `params` and `MACs` used by your model.

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> model_statistics --config configs/runs/path/to/config
```

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python -m lightning_template.tools.models.model_statistics --config configs/runs/path/to/config
```
