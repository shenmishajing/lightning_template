import thop
from lightning.pytorch.trainer.states import TrainerFn, TrainerStatus
from lightning.pytorch.utilities import rank_zero_warn

from lightning_template.utils import LightningCLI


def main():
    cli = LightningCLI(run=False)
    trainer, model, datamodule = cli.trainer, cli.model, cli.datamodule
    if hasattr(datamodule, "data_loader_config"):
        datamodule.data_loader_config["batch_size"] = 1
    else:
        rank_zero_warn(
            "Can not find data_loader_config in datamodule,"
            "you must ensure batch_size == 1 by yourself."
        )

    trainer.state.fn = TrainerFn.VALIDATING
    trainer.state.status = TrainerStatus.RUNNING
    trainer.validating = True

    trainer._data_connector.attach_data(model, datamodule=datamodule)

    # attach model to the training type plugin
    trainer.training_type_plugin.connect(model)

    # hook
    trainer._data_connector.prepare_data()
    trainer._callback_connector._attach_model_callbacks()

    # ----------------------------
    # SET UP TRAINING
    # ----------------------------
    trainer.call_hook("on_before_accelerator_backend_setup")
    trainer.accelerator.setup_environment()
    trainer._call_setup_hook()  # allow user to setup lightning_module in accelerator environment

    trainer._call_configure_sharded_model()  # allow user to setup in model sharded environment
    trainer.accelerator.setup(trainer)

    # plugin will setup fitting (e.g. ddp will launch child processes)
    trainer._pre_dispatch()

    trainer.reset_val_dataloader(model)

    dataloader = trainer.training_type_plugin.process_dataloader(
        trainer.val_dataloaders
    )[0]
    batch = None
    while batch is None:
        batch = next(iter(dataloader))

    batch = trainer.accelerator.batch_to_device(batch)

    total_ops, total_params = thop.profile(model, (batch,), verbose=True)
    total_ops, total_params = thop.clever_format([total_ops, total_params])
    print(f"total ops: {total_ops} total params: {total_params}")


if __name__ == "__main__":
    main()
