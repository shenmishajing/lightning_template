# Lightning Template

Lightning Template is a generic project template lib based on [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) for [project-template](https://github.com/shenmishajing/project_template) with the following features:

- All features from [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) and [lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html). Especially, the experiment manager feature, auto-implementing multi-node, multi-device, multi-accelerator support, etc.
- Powerful [deep update](configs/deep_update.md) feature for config file inherit to manage your config files more hierarchically, see also [recommend structure](configs/config_file_structure.md).
- Multi and complex optimizers and lr_scheduler from CLI config support, see [doc](core/optimizer_config.md).
- Cross-validation support with only one argument you have to change, see [doc](core/trainer.md).
- Hyperparameters tuning via Wandb, see the [doc](configs/argument_parsers/json_file.md) and [wandb sweep](https://docs.wandb.ai/guides/sweeps).
- Powerful and flexible LightningModule and LightningDataModule base class.
- Useful auto lr finder and auto batch size scaler, see [doc](tools/cli.md).

````{grid} 2
:gutter: 3

  ```{grid-item-card} Installation Guides

  Get started by installing Lightning Template.

  +++

    ```{button-ref} get_started/installation
    :expand:
    :color: secondary
    :click-parent:

    To the installation guides
    ```
  ```

  ```{grid-item-card} Usage Guides

  Details of the core features provide by Lightning Template.

  +++

    ```{button-ref} get_started/usage
    :expand:
    :color: secondary
    :click-parent:

    To the Usage guides
    ```
  ```

  ```{grid-item-card} API Reference Guides

  References of the API provided by Lightning Template.

  +++

    ```{button-ref} autoapi/lightning_template/index
    :expand:
    :color: secondary
    :click-parent:

    To the API Reference guides
    ```
  ```

  ```{grid-item-card} Contribution Guides

  Helpful instruction to develop Lightning Template.

  +++

    ```{button-ref} get_started/contribution
    :expand:
    :color: secondary
    :click-parent:

    To the Contribution guides
    ```
  ```
````

```{toctree}
:hidden:
:caption: Get Started

get_started/installation
get_started/usage
get_started/contribution
get_started/changelog
```

```{toctree}
:hidden:
:caption: Core

core/model
core/dataset
core/optimizer_config
core/trainer
```

```{toctree}
:hidden:
:caption: Configuration

configs/deep_update
configs/config_file_structure
configs/argument_parsers/index
```

```{toctree}
:hidden:
:caption: Tools

tools/cli
```

```{toctree}
:hidden:
:caption: API References

autoapi/lightning_template/index
```
